## Python

Multisession development package. Requires **Python 3.12+**.

### Setup

1. Create and activate a virtual environment (conda, mamba, venv, etc.):

```bash
mamba create -n multisession python=3.12 pip
mamba activate multisession
```

2. From the repository root, install dependencies (including the custom GTSAM wheel in `environment/`):

```bash
pip install -r requirements.txt
```

The GTSAM wheel is a pre-built, portable binary built from the Docker setup in `environment/`. It is not available on PyPI. If you rebuild the wheel, update the filename in `requirements.txt` to match the new file in `environment/`. See `environment/README.md` for how to build it.

3. Install this package in editable mode so local imports resolve:

```bash
pip install -e .
```

4. Verify GTSAM installed correctly:

```bash
python -c "import gtsam; print(gtsam.__file__)"
```

<!-- ### VS Code / Cursor type hints for GTSAM

The custom GTSAM wheel does not ship type stubs. Stub files live in `environment/typings/`. The workspace `.vscode/settings.json` configures Pylance to use them:

```json
{
    "python.analysis.stubPath": "./environment/typings",
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.extraPaths": ["."]
}
```

- `stubPath` — tells Pylance where to find `.pyi` stubs for `gtsam`.
- `extraPaths` — helps resolve imports from the repo root (e.g. `multiSession`, `common`).
- `typeCheckingMode` — enables basic static checking; set to `"off"` if you only want completions without diagnostics.

Open the repo root as the workspace folder so these relative paths resolve. Reload the window after changing settings if completions do not appear immediately. -->

### Notebooks

To run `tests/Tests.ipynb` or `sparcification/testNFR.ipynb`, install Jupyter in your environment:

```bash
pip install jupyter ipykernel ipympl
```

You may need to reload the window or restart the kernel.

## Running the Multi-Session Test

`tests/MultiSessionTest.py` is the main entry point. It streams a rosbag through KISS-ICP odometry, feeds keyframes into a `SessionManager` (a fixed-lag pose-graph smoother with loop closure), and optionally localizes against a previously saved session ("prior map").

> **Note:** near the bottom of `main()` there is a block under `# Hardcode values for testing purposes #` that overwrites whatever you pass on the CLI (dataset, missions, policy, load/save dirs, etc.). Comment that block out (or set the values to `None`/defaults) if you want your CLI flags to actually take effect.

### Basic usage

```bash
python -m tests.MultiSessionTest -d nc -m quad-easy -p INLIER_RATIO -v mpl
```

or, since it's also directly executable:

```bash
python tests/MultiSessionTest.py -d nc -m quad-easy quad-medium -p RELATIVE_COVARIANCE
```

Passing multiple `-m/--missions` runs them **sequentially as a chain**: the first mission starts a fresh session (or loads `--load_session_dir` if given), and each subsequent mission's session is seeded by loading the previous mission's saved output — building up a multi-session prior map over the sequence.

### CLI arguments

| Flag | Description |
| --- | --- |
| `-d`, `--dataset` | Dataset to use: `nc` (Newer College, default), `nclt` (North Campus Long-Term), or `urban` (BYU urban dataset). |
| `-m`, `--missions` | One or more mission/sequence names to run, in order (see [Datasets](#datasets) below for valid names per dataset). |
| `--prior_update_method` | How/when the prior map is updated with new session data: `0` = only at the end of the session ("post"), `1` = when transitioning from mapping to localization ("incremental"), `2` = continuously whenever in mapping mode ("continuous"). Defaults to `Params.priorUpdateMethod` (`0`). |
| `-l`, `--load_session_dir` | Directory of a previously saved session to load as the prior map for the first mission in the chain (see [Save / load session format](#save--load-session-format)). Omit to start from scratch. |
| `-s`, `--save_session_dir` | Directory to save the resulting session to. If omitted, a directory is auto-generated under `results/` from the dataset, transition-boundary type, prior-update method, and mission name (a numeric suffix is appended if it already exists). |
| `-v`, `--viz` | Visualization mode: `none`, `mpl` (matplotlib, live 2D graph plot), `o3d` (Open3D, live 3D map), or `both`. Overrides the module-level `ENABLE_MATPLOTLIB` / `ENABLE_OPEN3D` flags at the top of the file when set. |
| `-e`, `--evaluate_decision_types` | Run the same session multiple times, once per listed `DecisionType`, to compare policies side by side (each is saved to its own `<save_session_dir>/<DecisionType>/` subfolder). Overrides `-p/--policy` when set. |
| `-p`, `--policy` | Which loop-closure decision policy to use for deciding whether a new keyframe should localize against the prior map or extend the current map (ignored if `-e` is set). One of `NONE`, `DISTANCE`, `OVERLAP`, `INLIER_RATIO`, `INLIER_RMSE`, `TWO_STEP_VALIDATION`, `TWO_OPTION`, `RELATIVE_COVARIANCE`. See [Decision policies](#decision-policies) below. |
| `-t`, `--transition_check` | Enables an extra check (determinant of the information matrix at the first/last mapping nodes) to keep the localization/mapping boundary well-constrained before committing the transition. |

### Decision policies

The policy chosen with `-p`/`--policy` (or swept with `-e`) determines *when* a session should register against the prior map instead of continuing to build its own map, and each has its own tunable sub-config under `Params` (`distancePolicy`, `overlapPolicy`, `inlierRatioPolicy`, `inlierRMSEPolicy`, `twoStepConnectivityPolicy`, `relativeCovariancePolicy` — see the `Params` table below):

| Policy | Category | Idea |
| --- | --- | --- |
| `NONE` | — | No decision-making; always localize against the prior. |
| `DISTANCE` | Geometric | Trigger based on proximity (`distancePolicy.minDistance`) to the prior map. |
| `OVERLAP` | Geometric | Trigger based on point-cloud overlap ratio (`overlapPolicy.overlapThreshold`) with the prior map. |
| `INLIER_RATIO` | Registration | Trigger based on ICP fitness/inlier ratio (`inlierRatioPolicy.fitnessThreshold`) between scan and prior map. |
| `INLIER_RMSE` | Registration | Trigger based on ICP inlier RMSE (`inlierRMSEPolicy.inlierRMSEThreshold`). |
| `TWO_STEP_VALIDATION` | Graph-theoretic | Uses a sliding-window weighted pose graph (Laplacian / algebraic connectivity, `twoStepConnectivityPolicy`) to validate loop closures in two steps. |
| `TWO_OPTION` | Graph-theoretic | Variant of the graph-theoretic policy (see `policy/graphTheoretic.py`). |
| `RELATIVE_COVARIANCE` | Probabilistic | Trigger based on the relative marginal covariance / log-determinant (`relativeCovariancePolicy`) between current and prior estimates. |

### Key `Params` fields

`Params` (in `params/params.py`) holds the full session configuration; most of it is set for you by the CLI args above, but the following are useful to know when reading or editing the code/defaults:

| Field | Meaning |
| --- | --- |
| `fixedLagKeepFrames` | Number of most-recent keyframes kept in the fixed-lag smoother's active window before marginalization. |
| `fixedLagMarg` | Marginalization strategy for frames leaving the window (`FixedLagMarginalizationType`: `DENSE`, `APPROX_PRIOR`, `UNARY`, `PRIOR_ON_ANCHOR_ALL`, `PRIOR_ON_ANCHOR_FIRST`, `THROW_AWAY_BETWEEN`). |
| `linearFactorMaxNodes` | Max nodes considered when linearizing certain factors. |
| `currSessionSymbolChr` / `addToPriorSymbolChr` / `priorSessionSymbolChrs` | GTSAM symbol character prefixes used to distinguish the current session, nodes being added to the prior, and prior sessions' own symbols in the factor graph. |
| `rekeyPrior` | Whether prior-session keys are rekeyed to avoid collisions when merging into the current graph. |
| `policyType` | The `DecisionType` in use (set via `-p`/`-e`). |
| `enableTransitionInformationCheck` | Same as `-t`/`--transition_check`. |
| `dataset` / `mission` | Dataset name and mission/sequence name (set via `-d`/`-m`). |
| `bagFile` / `gtFile` | Paths to the rosbag and ground-truth file for the mission (resolved in `main()` from the dataset's hardcoded data directory). |
| `priorUpdateMethod` | Same as `--prior_update_method`. |
| `loadSessionDir` / `saveSessionDir` | Same as `-l`/`-s`. |
| `kissICPConfigFile` | Path to the KISS-ICP odometry YAML config (`params/kiss_icp.yaml`). |
| `refineOdometry` | If true, runs an additional scan-match refinement on top of KISS-ICP odometry. |
| `useGps` / `gpsCovThreshold` / `gpsDistThreshold` | Whether to fuse GPS factors, and the covariance/distance thresholds used to gate them. |
| `icp.voxelSize` | Downsampling voxel size for ICP point clouds (`None` disables downsampling). |
| `loopClosure.*` | Loop-closure search/registration tuning: search-window sizes (`intraNearKeyFrames`, `interNearKeyFramesCurrent`, `interNearKeyFramesPrior`), `voxelSize`, `fitnessThreshold`, `rmseThreshold`, `maxCorrespondenceDist`, and debug visualization flags (`showScanMatchResult`, `showResultSkipFrames`). |

### Datasets

Datasets are selected with `-d`/`--dataset`. The root directory for each dataset is currently **hardcoded per-machine** in `main()` in `tests/MultiSessionTest.py` (around the `if args.dataset == "nc": ...` block) — edit those paths to match your machine before running:

```python
nc_data_dir = Path("/home/benb/data_sets/NewerCollege/ros2")
urban_data_dir = Path("/home/benb/data_byu/bags/urban")
nclt_data_dir = Path("/home/benb/data_sets/NCLT")           # or .../NCLT/nclt2ros for most missions
```

For each mission, the code expects:

- **A ROS2 bag** (a `ros2 bag` directory, read via `rosbags.highlevel.AnyReader`) containing a recognized lidar topic and, optionally, a GPS topic:
  - Lidar topics recognized: `/os_cloud_node/points`, `/os1_cloud_node/points`, `/ouster_top/points`, `/ouster/points`, `/velodyne_points`.
  - GPS topics recognized: `/navsat/fix`, `/inertial_sense/gps1/NavSatFix`.
- **A ground-truth file** in TUM-like CSV format: either 7 columns (`t, x, y, z, roll, pitch, yaw`) or 8 columns (`t, x, y, z, q_x, q_y, q_z, q_w`), comma- or space-delimited.

Per-dataset layout expected by `main()`:

| Dataset (`-d`) | Bag path | Ground-truth path |
| --- | --- | --- |
| `nc` (Newer College) | `<nc_data_dir>/<mission>` | `<nc_data_dir>/ground_truth_tum/gt-nc-<mission>.csv` |
| `urban` (BYU) | `<urban_data_dir>/<mission>` | `<urban_data_dir>/groundTruth/<mission>_gt/optimized_poses.txt` |
| `nclt` (`2012-01-08_bin` mission) | `<nclt_data_dir>/2012-01-08/2012-01-08_bin_ros2` | `<nclt_data_dir>/2012-01-08/groundtruth_2012-01-08.csv` |
| `nclt` (other missions) | `<nclt_data_dir>/nclt2ros/rosbags/<mission>/ros2_bag` | `<nclt_data_dir>/nclt2ros/raw_data/<mission>/ground_truth/groundtruth_<mission>.csv` |

If a bag or ground-truth file is missing, `main()` prints a warning but still attempts the run (it will fail once the bag reader can't find the path).

For `nclt`, a GPS reference origin (`lat=42.293227, lon=-83.709657, alt=270`) is set automatically via `SessionManager.SetGpsReference`, and the default ICP `maxCorrespondingDistance` is widened to `5.0` (vs. `0.5` for other datasets) unless loading an existing session.

### Save / load session format

`SessionManager.SaveSession` / `LoadSession` persist a session as a directory containing (at minimum) `graph.gtsam`, `values.gtsam`, `odom_tum.txt`, `timestamps.txt`, plus saved point-cloud scans and final-graph plots. When `--load_session_dir`/`--save_session_dir` (or `priorUpdateMethod == 2`) is combined with `-e/--evaluate_decision_types`, sessions are additionally nested one level under `<dir>/<DecisionType>/` per policy being evaluated.

### Output

Unless `-s/--save_session_dir` is given, results are written under:

```
results/<dirAfterResults>/<dataset>/<wellConstrained|exact>/<post|incremental|continuous>/<mission>/
```

where `<dirAfterResults>` is set in `main()` (default in the current hardcoded block is `"twoStepEval"`), `wellConstrained`/`exact` reflects `-t/--transition_check`, and `post`/`incremental`/`continuous` reflects `--prior_update_method`. Each run directory includes the saved session, `final_graph.png`, `final_graph_with_covariances.png`, and (if ground truth was available) `final_graph_with_ground_truth.png`.
