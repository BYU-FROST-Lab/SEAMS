#!/usr/bin/env python3
"""3D opening-figure visualization: factor graph + voxelized point clouds.

Colors match the 2D overlay in ``evaluations/opening_figure.ipynb``:
  Prior (A)            C1 / tab:orange
  Mapping (B)  tab:green
  Localization (c, excluding boundary keys)  tab:blue
  Boundary     tab:red

The plot shows only the graph, the point clouds, and a legend — no grid,
axes, ticks, or labels.

Notebook::

    from evaluations.opening_figure_3d import (
        group_keys_from_session,
        build_colored_clouds,
        plot_opening_figure_3d,
    )

    groups = group_keys_from_session(session, boundaryKeyRange)
    clouds = build_colored_clouds(session.scans, values, groups)
    fig, ax = plot_opening_figure_3d(graph, values, clouds, groups)

Standalone::

    python -m evaluations.opening_figure_3d
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import gtsam
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

S = gtsam.Symbol

DEFAULT_SESSION_DIR = Path(
    "results/transitionBoundaryEval/nc/wellConstrained/quad-easy_startMapping/DISTANCE"
)
DEFAULT_BOUNDARY_RANGE = range(44, 54)
VOXEL_SIZE = 0.5

# Draw order: largest / background group first so later groups sit on top.
GROUP_ORDER = ("A", "c", "Boundary", "B")
# Legend order is independent of draw order (top → bottom, or left → right if ncol>1).
LEGEND_ORDER = ("A", "c", "Boundary", "B")
GROUP_COLORS = {
    "A": "C1",
    "B": "tab:green",
    "c": "tab:blue",
    "Boundary": "tab:red",
}
GROUP_LABELS = {
    "A": "Prior",
    "B": "Mapping",
    "c": "Localization",
    "Boundary": "Decision Region",
}
INTER_SESSION_EDGE_COLOR = "tab:olive"


def group_keys_from_session(
    session,
    boundary_key_range: list[int] | set[int],
) -> dict[str, set[int]]:
    """Split session keys into A / mapping / localization / boundary groups."""
    boundary_keys = set(boundary_key_range)
    mapping_keys = set(session.sessionKeys.get("B", []))
    localization_keys = set(session.sessionKeys.get("c", [])) - boundary_keys
    prior_keys = set(session.sessionKeys.get("A", []))
    return {
        "A": prior_keys,
        "B": mapping_keys,
        "c": localization_keys,
        "Boundary": boundary_keys,
    }


def key_group(key: int, groups: dict[str, set[int]]) -> str | None:
    for name in ("Boundary", "B", "c", "A"):
        if key in groups[name]:
            return name
    return None


def key_color(key: int, groups: dict[str, set[int]]) -> str:
    name = key_group(key, groups)
    return GROUP_COLORS[name] if name is not None else "0.5"


def remove_long_intra_session_loops(
    graph: gtsam.NonlinearFactorGraph,
    session_keys: list[int],
    max_translation: float = 10.0,
) -> gtsam.NonlinearFactorGraph:
    """Remove intra-session BetweenFactorPose3 edges longer than *max_translation*."""
    session_set = set(session_keys)
    for i in range(graph.size()):
        factor = graph.at(i)
        if not isinstance(factor, gtsam.BetweenFactorPose3):
            continue
        k1, k2 = factor.keys()[0], factor.keys()[1]
        if k1 not in session_set or k2 not in session_set:
            continue
        dist = np.linalg.norm(factor.measured().translation())
        if dist > max_translation:
            graph.remove(i)
            print(
                f"Removed factor {i} between {S(k1).string()} and {S(k2).string()} "
                f"with distance {dist}"
            )
    return graph


def build_colored_clouds(
    pcds: dict[int, o3d.geometry.PointCloud],
    values: gtsam.Values,
    groups: dict[str, set[int]],
    voxel_size: float = VOXEL_SIZE,
) -> dict[str, o3d.geometry.PointCloud]:
    """Merge, voxelize, and paint each group's scans in the world frame."""
    colored: dict[str, o3d.geometry.PointCloud] = {}
    for name in GROUP_ORDER:
        keys = groups[name]
        merged = o3d.geometry.PointCloud()
        n_used = 0
        for key in keys:
            if key not in pcds or not values.exists(key):
                continue
            if voxel_size:
                pcd = pcds[key].voxel_down_sample(voxel_size)
            else:
                pcd = copy.deepcopy(pcds[key])
            pcd.transform(values.atPose3(key).matrix())
            merged += pcd
            n_used += 1
        if voxel_size and len(merged.points) > 0:
            merged = merged.voxel_down_sample(voxel_size)
        if len(merged.points) > 0:
            merged.paint_uniform_color(to_rgb(GROUP_COLORS[name]))
        print(f"{GROUP_LABELS[name]}: {n_used} scans -> {len(merged.points)} points")
        colored[name] = merged
    return colored


def _factor_edge_color(factor, groups: dict[str, set[int]]) -> str:
    keys = list(factor.keys())
    keyset = set(keys)
    if not keyset:
        return INTER_SESSION_EDGE_COLOR
    if keyset <= groups["Boundary"]:
        return GROUP_COLORS["Boundary"]
    if keyset <= groups["B"]:
        return GROUP_COLORS["B"]
    if keyset <= (groups["c"] | groups["Boundary"]):
        return GROUP_COLORS["c"]
    if keyset <= groups["A"]:
        if len(keys) == 2 and abs(keys[0] - keys[1]) == 1:
            return GROUP_COLORS["A"]
        return INTER_SESSION_EDGE_COLOR
    if isinstance(factor, gtsam.LinearContainerFactor):
        return "tab:purple"
    return INTER_SESSION_EDGE_COLOR


def pose_xyz(
    values: gtsam.Values,
    key: int,
    groups: dict[str, set[int]],
    z_offsets: dict[str, float] | None = None,
) -> tuple[float, float, float]:
    """World-frame node position, with an optional per-group +z drawing offset."""
    p = values.atPose3(key)
    z = p.z()
    name = key_group(key, groups)
    if z_offsets and name in z_offsets:
        z += z_offsets[name]
    return (p.x(), p.y(), z)


def collect_graph_geometry(
    graph: gtsam.NonlinearFactorGraph,
    values: gtsam.Values,
    groups: dict[str, set[int]],
    z_offsets: dict[str, float] | None = None,
) -> tuple[list, list[str], np.ndarray, list[str]]:
    """Return factor segments, segment colors, node xyz, and node colors."""
    segments: list = []
    seg_colors: list[str] = []
    for i in range(graph.size()):
        factor = graph.at(i)
        if factor is None:
            continue
        keys = [k for k in factor.keys() if values.exists(k)]
        if len(keys) < 2:
            continue
        pts = [pose_xyz(values, k, groups, z_offsets) for k in keys]
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        cz = float(np.mean([p[2] for p in pts]))
        color = _factor_edge_color(factor, groups)
        for pt in pts:
            segments.append([pt, (cx, cy, cz)])
            seg_colors.append(color)

    node_xyz = []
    node_colors = []
    for k in values.keys():
        node_xyz.append(list(pose_xyz(values, k, groups, z_offsets)))
        node_colors.append(key_color(k, groups))
    return segments, seg_colors, np.asarray(node_xyz), node_colors


def hide_axis_decorations(ax) -> None:
    """Leave only the data and legend — no box, grid, ticks, or labels."""
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_axis_off()
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_alpha(0.0)
        axis.line.set_color((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.0
        axis._axinfo["axisline"]["color"] = (1.0, 1.0, 1.0, 0.0)
        axis._axinfo["axisline"]["linewidth"] = 0.0


def scene_limits(
    node_xyz: np.ndarray,
    cloud_points: list[np.ndarray],
    percentile: tuple[float, float] = (0.2, 99.8),
    pad_frac: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Frame the scene from poses and clouds without AABB-cropping the scans."""
    parts = []
    if len(node_xyz):
        parts.append(np.asarray(node_xyz))
    for pts in cloud_points:
        if pts.size:
            parts.append(pts)
    if not parts:
        return np.zeros(3), np.ones(3)
    pts = np.vstack(parts)
    lo = np.percentile(pts, percentile[0], axis=0).astype(float)
    hi = np.percentile(pts, percentile[1], axis=0).astype(float)
    span = np.maximum(hi - lo, 1e-6)
    lo -= pad_frac * span
    hi += pad_frac * span
    return lo, hi


def set_metric_aspect(ax, mins: np.ndarray, maxs: np.ndarray) -> None:
    """Fill the axes with the data while keeping metres equal on x/y/z."""
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ranges = np.maximum(maxs - mins, 1e-6)
    try:
        ax.set_box_aspect(tuple(ranges), zoom=1.45)
    except TypeError:
        ax.set_box_aspect(tuple(ranges))
        if hasattr(ax, "dist"):
            ax.dist = 7.5
    except AttributeError:
        pass


def make_legend_handles(groups: dict[str, set[int]] | None = None) -> list[Line2D]:
    """Legend markers matching the 3D opening figure, in ``LEGEND_ORDER``."""
    names = [
        name for name in LEGEND_ORDER
        if groups is None or name in groups
    ]
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=GROUP_COLORS[name],
            markeredgecolor="k",
            markeredgewidth=0.3,
            markersize=8,
            linestyle="None",
            label=GROUP_LABELS[name],
        )
        for name in names
    ]


def save_opening_legend(
    path: str | Path,
    groups: dict[str, set[int]] | None = None,
    ncol: int = 1,
) -> Path:
    """Save only the legend, cropped tightly to the legend box."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handles = make_legend_handles(groups)
    labels = [h.get_label() for h in handles]
    print("Legend order:", labels)
    fig = plt.figure(figsize=(3.0, 2.0))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    legend = ax.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="0.85",
        framealpha=1.0,
        borderpad=0.4,
    )
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, dpi=300, bbox_inches=bbox, pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(f"Saved legend {path}")
    return path


def plot_opening_figure_3d(
    graph: gtsam.NonlinearFactorGraph,
    values: gtsam.Values,
    clouds: dict[str, o3d.geometry.PointCloud],
    groups: dict[str, set[int]],
    ax=None,
    elev: float = 28.0,
    azim: float = -70.0,
    cloud_point_size: float = 0.6,
    cloud_alpha: dict[str, float] | float | None = None,
    node_size: float = 28.0,
    edge_linewidth: float = 1.2,
    figsize: tuple[float, float] = (8.0, 6.5),
    localization_z_offset: float = 2.0,
    show_legend: bool = True,
    save_path: str | Path | None = None,
    legend_path: str | Path | None = None,
):
    """Draw voxelized clouds + factor graph with no axis decorations."""
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    if cloud_alpha is None:
        cloud_alpha = {"A": 0.22, "B": 0.40, "c": 0.40, "Boundary": 0.50}
    elif not isinstance(cloud_alpha, dict):
        cloud_alpha = {name: float(cloud_alpha) for name in GROUP_ORDER}

    z_offsets = {"c": localization_z_offset} if localization_z_offset else {}

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass

    segments, seg_colors, node_xyz, node_colors = collect_graph_geometry(
        graph, values, groups, z_offsets=z_offsets
    )
    raw_clouds = {
        name: np.asarray(cloud.points) if cloud is not None and len(cloud.points) else np.empty((0, 3))
        for name, cloud in ((n, clouds.get(n)) for n in GROUP_ORDER)
    }
    mins, maxs = scene_limits(node_xyz, list(raw_clouds.values()))

    for name in GROUP_ORDER:
        pts = raw_clouds[name]
        if pts.size == 0:
            continue
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=cloud_point_size if name == "A" else max(cloud_point_size, 1.0),
            c=GROUP_COLORS[name],
            alpha=cloud_alpha[name],
            linewidths=0,
            rasterized=True,
            depthshade=False,
            zorder=1,
            clip_on=False,
        )

    if segments:
        lc = Line3DCollection(
            segments,
            colors=seg_colors,
            linewidths=edge_linewidth,
            alpha=0.95,
            zorder=3,
        )
        lc.set_clip_on(False)
        ax.add_collection3d(lc)
    if len(node_xyz):
        ax.scatter(
            node_xyz[:, 0],
            node_xyz[:, 1],
            node_xyz[:, 2],
            s=node_size,
            c=node_colors,
            depthshade=False,
            linewidths=0.35,
            edgecolors="k",
            zorder=4,
            clip_on=False,
        )

    set_metric_aspect(ax, mins, maxs)

    hide_axis_decorations(ax)
    ax.view_init(elev=elev, azim=azim)

    if show_legend:
        handles = make_legend_handles(groups)
        # Use Figure.legend so 3D scatter artists cannot override handle order.
        fig.legend(
            handles=handles,
            labels=[h.get_label() for h in handles],
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="0.85",
            framealpha=0.95,
            borderpad=0.4,
        )

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        print(f"Saved {save_path}")
    if legend_path is not None:
        save_opening_legend(legend_path, groups=groups)
    return fig, ax


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        default=DEFAULT_SESSION_DIR,
        help="Session directory containing graph.gtsam / values.gtsam / scans/",
    )
    parser.add_argument("--voxel-size", type=float, default=VOXEL_SIZE)
    parser.add_argument("--elev", type=float, default=28.0)
    parser.add_argument("--azim", type=float, default=-70.0)
    parser.add_argument("-s", "--save", type=Path, default=None, help="Optional output image path")
    args = parser.parse_args(argv)

    # Local imports so the notebook can use this module without SessionManager
    # unless it is running standalone.
    from multiSession.SessionManager import SessionManager

    directory = args.directory
    if not directory.is_absolute():
        directory = (Path(__file__).resolve().parents[1] / directory).resolve()

    print(f"Loading session from {directory}")
    session = SessionManager(directory)
    graph = session.graph
    values = session.values

    if "A" in session.sessionKeys:
        remove_long_intra_session_loops(graph, session.sessionKeys["A"], max_translation=10.0)

    boundary_keys = [S("c", i).key() for i in DEFAULT_BOUNDARY_RANGE]
    groups = group_keys_from_session(session, boundary_keys)
    clouds = build_colored_clouds(session.scans, values, groups, voxel_size=args.voxel_size)
    plot_opening_figure_3d(
        graph,
        values,
        clouds,
        groups,
        elev=args.elev,
        azim=args.azim,
        save_path=args.save,
    )
    plt.show()


if __name__ == "__main__":
    main()
