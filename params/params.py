from dataclasses import dataclass, field
from common.customTypes import DecisionType, FixedLagMarginalizationType
from pathlib import Path

# ICP params
@dataclass
class ICP:
    voxelSize: float | None = 0.4   # Either > 0.0 or None. If None, no downsampling is performed.

# Loop Closure params
@dataclass
class LoopClosure:
    noiseScale: float = 1
    prevFrameSkip: int = 50
    distMax: int = 40
    skipFrames: int = 0
    intraNearKeyFrames: int = 5     # Same session
    interNearKeyFramesCurrent: int = 0     # >=0 Between session. Number of key frames to use in current window in addition to the current pose for registration to the prior session.
    interNearKeyFramesPrior: int = 5     # >=0 Between session. Number of key frames on each side of the current pose to use for registration to the prior session.
    voxelSize: float | None = 0.4
    fitnessThreshold: float = 0.5
    rmseThreshold: float = 0.3
    maxCorrespondenceDist: float = 0.5
    showScanMatchResult: int = 0    # 0 = no visualization, 1 = visualize initial alignment, 2 = visualize final alignment, 3 = visualize both
    showResultSkipFrames: int = 0   # 0 = visualize every frame, 1 = visualize every other frame, 2 = visualize every third frame, etc.
    debug: bool = True

# Geometry policy
@dataclass
class DistancePolicy:
    minDistance: float | int = 37
    debug: bool = True

# Overlap policy
@dataclass
class OverlapPolicy:
    overlapThreshold: float = 0.7   # [0, 1]
    debug: bool = False

# Registration policy
@dataclass
class InlierRatioPolicy:
    fitnessThreshold: float = 0.70
    fitnessHysteresisMargin: float = 0.0
    debug: bool = False

@dataclass
class InlierRMSEPolicy:
    inlierRMSEThreshold: float = 0.24
    inlierRMSEHysteresisMargin: float = 0.0
    minCorrespondenceSet: int = 1000
    vizScanMatch: bool = False
    debug: bool = False

# Graph Theoretic
@dataclass
class TwoStepConnectivityPolicy:
    # Sliding window for the WeightedPoseGraph used to compute the Laplacian,
    # d-bar, and lambda_2. Must be <= Params.fixedLagKeepFrames.
    graphWindowLength: int = 12
    # Sliding window for the d-bar mu/sigma moving-average history.
    movingAverageLength: int = 5
    # Floor on sigma as a fraction of mu so a perfectly stable d-bar does not
    # treat every numerical wiggle as a 2/4-sigma drop (arXiv:2602.17226 §III-B2).
    minRelativeSigma: float = 0.01
    maxConnections: int = 1 # maximum loop closure connections for two step validation
    # twoOption: bool = False
    showDLplot: bool = False
    debug: bool = False

# Probabilistic
@dataclass
class RelativeCovariancePolicy:
    threshold: list[float] = field(
        default_factory=lambda: [1., 1., 1.]    # [x, y, z, det]
        )
    logDetThreshold: float = -38.0
    debug: bool = True


@dataclass
class Params:
    ##################################################################
    # General Multi-Session Params
    ##################################################################

    # Fixed Lag Smoother Params
    fixedLagKeepFrames: int = 80
    fixedLagMarg: FixedLagMarginalizationType = FixedLagMarginalizationType.APPROX_PRIOR
    # fixedLagMarg: FixedLagMarginalizationType = FixedLagMarginalizationType.DENSE
    linearFactorMaxNodes: int = 2       # 2 = Binary factor

    # Symbol values
    currSessionSymbolChr: str = 'c'
    addToPriorSymbolChr: str = 'A'
    priorSessionSymbolChrs: list[str] = field(default_factory=list)
    rekeyPrior: bool = True
    autoRename: bool = True
        
    # Decision params
    policyType: DecisionType = DecisionType.INLIER_RATIO
    enableTransitionInformationCheck: bool = True

    # Current session params
    dataset: str = "nc"
    mission: str = "quad-easy"
    dataDir: Path = Path("/home/benb/data_sets/NewerCollege/ros2")
    bagFile: Path | None = dataDir / mission
    gtFile: Path | None = dataDir / f"ground_truth_tum/gt-nc-{mission}.csv"

    # 0: Only add to prior at end of session
    # 1: add to prior when transitioning from mapping to localization 
    # 2: add to prior whenever in mapping mode
    priorUpdateMethod: int = 0

    # Save session params
    moduleDir: Path = Path(__file__).parents[1]
    loadSessionDir: Path |None = None
    saveSessionDir: Path = moduleDir / f"results/test/{dataset}/{mission}"
    # Global point cloud store shared by all sessions: {scansRootDir}/{dataset}/{mission}/{timestamp_us}.pcd
    scansRootDir: Path = moduleDir / "results/scans"

    # Odometry
    kissICPConfigFile: Path = moduleDir / "params/kiss_icp.yaml"
    refineOdometry: bool = False    # Refine the odometry using an additional scan match after kiss-icp

    # Boundary Thresholds: Positive -> increasing covariance, Negative -> decreasing covariance
    boundaryLoc2Map: float = 0.1
    boundaryMap2Loc: float = 0.1
    boundaryMap2LocNearZero: float = 0.01

    # GPS
    useGps: bool = False
    gpsCovThreshold: float = 2.0    # meters
    gpsDistThreshold: float = 10.0  # meters
    
    ##################################################################
    # Decision Policy Params
    ##################################################################

    # Other dataclasses (other structs)
    icp: ICP = field(default_factory=ICP)
    loopClosure: LoopClosure = field(default_factory=LoopClosure)
    distancePolicy: DistancePolicy = field(default_factory=DistancePolicy)
    overlapPolicy: OverlapPolicy = field(default_factory=OverlapPolicy)
    inlierRatioPolicy: InlierRatioPolicy = field(default_factory=InlierRatioPolicy)
    inlierRMSEPolicy: InlierRMSEPolicy = field(default_factory=InlierRMSEPolicy)
    twoStepConnectivityPolicy: TwoStepConnectivityPolicy = field(default_factory=TwoStepConnectivityPolicy)
    relativeCovariancePolicy: RelativeCovariancePolicy = field(default_factory=RelativeCovariancePolicy)
