import gtsam
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


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


##############################################
# Parameters for the marginalization problem #
##############################################

num_prior_poses = 6
lag_window_size = 3

##############################################



prior_poses = list(range(num_prior_poses))

lag_poses = [0]


fig, ax = plt.subplots(6, 2, figsize=(FULL_WIDTH, TRIPLE_HEIGHT))

def draw_prior_factor(prior_pose: int | None, lag_pose: int | None, ax, color="tab:green"):
    if prior_pose is not None:
        ax.plot([prior_pose - 0.5, prior_pose], [0, 0], c=color, linestyle="-", zorder=0)
        ax.scatter(prior_pose - 0.5, 0, c=color, marker="s", s=3)
    if lag_pose is not None:
        ax.plot([lag_pose - 0.5, lag_pose], [1, 1], c=color, linestyle="-", zorder=0)
        ax.scatter(lag_pose - 0.5, 1, c=color, marker="s", s=3)

def draw_factor_between_poses(prior_poses: list, lag_poses: list, ax, color="tab:gray", linestyle="-", center_y=None):
    poses = [[x, 0] for x in prior_poses] + [[x, 1] for x in lag_poses]
    cx, cy = np.mean(poses, axis=0)
    if center_y is not None:
        cy = center_y
    for pose in poses:
        ax.plot([pose[0], cx], [pose[1], cy], c=color, linestyle=linestyle, zorder=0)
    ax.scatter(cx, cy, c=color, marker="s", s=3)

    return cx, cy

def draw_poses(prior_poses: list, poses_lag: list, ax):
    for pose in prior_poses:
        ax.scatter(pose, 0, c="tab:orange")
    for pose in poses_lag:
        ax.scatter(pose, 1, c="tab:blue")

def draw_odom_poses(poses_prior: list, poses_lag: list, ax,):
    draw_poses(poses_prior, poses_lag, ax)
    for i in range(len(poses_prior) - 1):
        draw_factor_between_poses([poses_prior[i], poses_prior[i + 1]], [], ax, color="tab:orange", linestyle="-")
    for i in range(len(poses_lag) - 1):
        draw_factor_between_poses([], [poses_lag[i], poses_lag[i + 1]], ax, color="tab:blue", linestyle="-")

def draw_loop_closures(poses_prior: list, poses_lag: list, ax):
    for pose in poses_lag:
        if pose in poses_prior:
            draw_factor_between_poses([pose], [pose], ax, color="tab:olive", linestyle="-")

# Set up plots with odometry and loop closures
for i in range(ax.shape[0]):
    draw_odom_poses(prior_poses, lag_poses, ax[i, 0])
    draw_odom_poses(prior_poses, lag_poses, ax[i, 1])
    draw_loop_closures(prior_poses, lag_poses, ax[i, 0])
    draw_loop_closures(prior_poses, lag_poses, ax[i, 1])

    if i < 3:
        draw_prior_factor(prior_poses[0], lag_poses[0], ax[i, 0])
        draw_prior_factor(prior_poses[0], lag_poses[0], ax[i, 1])
    else:
        draw_prior_factor(prior_poses[0], None, ax[i, 0])
        draw_prior_factor(prior_poses[0], None, ax[i, 1])

    # Show the positive x direction from the rightmost lag node in each panel.
    rightmost_lag_pose = max(lag_poses)
    for panel_ax in ax[i]:
        panel_ax.annotate(
            "",
            xy=(rightmost_lag_pose + 0.5, 1),
            xytext=(rightmost_lag_pose, 1),
            arrowprops={"arrowstyle": "->", "color": "tab:blue", "linewidth": 1.2},
            zorder=3,
        )

    if len(lag_poses) < lag_window_size:
        lag_poses.append(lag_poses[-1] + 1)
    else:
        lag_poses = [pose + 1 for pose in lag_poses]

    # Set up axis limits and labels
    ax[i, 0].set_xlim(-1, num_prior_poses)
    ax[i, 0].set_ylim(-1, 2)
    ax[i, 1].set_xlim(-1, num_prior_poses)
    ax[i, 1].set_ylim(-1, 2)

    ax[i, 0].set_xticks([])
    ax[i, 0].set_yticks([])
    ax[i, 1].set_xticks([])
    ax[i, 1].set_yticks([])

    ax[i, 0].set_frame_on(False)
    ax[i, 1].set_frame_on(False)

# Set axis labels for the subplots
ax[0, 0].set_xlabel("(a)")
ax[1, 0].set_xlabel("(b)")
ax[2, 0].set_xlabel("(c)")
ax[3, 0].set_xlabel("(d)")
ax[4, 0].set_xlabel("(e)")
ax[5, 0].set_xlabel("(f)")

ax[0, 1].set_xlabel("(g)")
ax[1, 1].set_xlabel("(h)")
ax[2, 1].set_xlabel("(i)")
ax[3, 1].set_xlabel("(j)")
ax[4, 1].set_xlabel("(k)")
ax[5, 1].set_xlabel("(l)")

# Exact Marginalization
for i in range(3):
    draw_factor_between_poses(list(range(i + 1)), [i + 1], ax[i + 3, 0], color="tab:purple", linestyle="-", center_y=0.5)
    cx, cy = draw_factor_between_poses([i], [i + 1], ax[i + 3, 1], color="tab:purple", linestyle="-")
    if i > 0:
        ax[i + 3, 1].plot([i-1, cx], [0, cy], color="tab:red", linestyle="--", zorder=0)

ax[0, 0].set_title("Exact Marginalization", fontweight="bold")
ax[0, 1].set_title("Proposed Approximation", fontweight="bold")

# Give the shared node and factor styles one key beneath the full figure.
legend_handles = [
    Line2D([], [], marker="o", linestyle="", color="tab:orange", label="Prior Node"),
    Line2D([0, 0.5, 1], [0, 0, 0], color="tab:orange", linestyle="-", marker="s",
           markevery=[1], label="Prior Odom. Factor"),
    Line2D([], [], marker="o", linestyle="", color="tab:blue", label="Current Node"),
    Line2D([0, 0.5, 1], [0, 0, 0], color="tab:blue", linestyle="-", marker="s",
           markevery=[1], label="Current Odom. Factor"),
    Line2D([0, 0.5, 1], [0, 0, 0], color="tab:olive", linestyle="-", marker="s",
           markevery=[1], label="Loop Closure Factor"),
    Line2D([0, 0.0, 1], [0, 0, 0], color="tab:green", linestyle="-", marker="s",
           markevery=None, label="Prior Factor"),
    Line2D([0, 0.5, 1], [0, 0, 0], color="tab:purple", linestyle="--", marker="s",
           markevery=[1], label="Marginal Factor"),
    Line2D([0, 0.5, 1], [0, 0, 0], color="tab:red", linestyle="--", label="Approximated Out"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, 0.0), frameon=False)

line = Line2D((0.5, 0.5), (0.1, 0.94), color="gray", linewidth=1.5, linestyle="-")
fig.add_artist(line)

# Keep the legend clear of the bottom row of subplots.
fig.tight_layout(rect=(0, 0.1, 1, 1))


# Save the figure
savePath = Path().home() / "Pictures/MastersThesis" / "marginalization_problem_plot.pdf"
fig.savefig(savePath, dpi=300)
print(f"Figure saved to {savePath}")

plt.show()
