from dataclasses import dataclass, field
from common.customTypes import DecisionType, FixedLagMarginalizationType
from pathlib import Path

# ICP params
@dataclass
class ICP:
    voxelSize: float | None = 0.1   # Either > 0.0 or None. If None, no downsampling is performed.

# Loop Closure params
@dataclass
class LoopClosure:
    noiseScale: float = 1
    prevFrameSkip: int = 50
    distMax: int = 20
    skipFrames: int = 0
    intraNearKeyFrames: int = 5     # Same session
    interNearKeyFramesPrior: int = 5     # >=0 Between session. Number of key frames on each side of the current pose to use for registration to the prior session.
    interNearKeyFramesCurrent: int = 0     # >=0 Between session. Number of key frames to use in current window in addition to the current pose for registration to the prior session.
    voxelSize: float | None = 0.4
    maxPossibleConnections: int = 1
    fitnessThreshold: float = 0.5
    showScanMatchResult: int = 0    # 0 = no visualization, 1 = visualize initial alignment, 2 = visualize final alignment, 3 = visualize both
    showResultSkipFrames: int = 0   # 0 = visualize every frame, 1 = visualize every other frame, 2 = visualize every third frame, etc.
    debug: bool = False

# Geometry policy
@dataclass
class DistancePolicy:
    minDistance: float | int = 10

# Overlap policy
@dataclass
class OverlapPolicy:
    overlapThreshold: float = 0.7   # [0, 1]
    debug: bool = False

# Registration policy
@dataclass
class FitnessPolicy:
    fitnessThreshold: float = 0.55
    fitnessHysteresisMargin: float = 0.0
    debug: bool = False

@dataclass
class InlierRMSEPolicy:
    inlierRMSEThreshold: float = 0.25
    inlierRMSEHysteresisMargin: float = 0.01
    debug: bool = False

# Graph Theoretic
@dataclass
class TwoStepConnectivityPolicy:
    movingAverageLength: int = 10
    # Floor on sigma as a fraction of mu so a perfectly stable d-bar does not
    # treat every numerical wiggle as a 2/4-sigma drop (arXiv:2602.17226 §III-B2).
    minRelativeSigma: float = 0.01
    showDLplot: bool = True
    debug: bool = True

# Probabilistic
@dataclass
class RelativeCovariancePolicy:
    threshold: list[float] = field(
        default_factory=lambda: [1., 1., 1.]    # [x, y, z, det]
        )
    debug: bool = False


@dataclass
class Params:
    ##################################################################
    # General Multi-Session Params
    ##################################################################

    # Fixed Lag Smoother Params
    fixedLagCurrentSession: bool = False
    fixedLagKeepFrames: int = 10
    fixedLagMarg: FixedLagMarginalizationType = FixedLagMarginalizationType.APPROX_PRIOR
    linearFactorMaxNodes: int = 2

    # Symbol values
    currSessionSymbolChr: str = 'c'
    addToPriorSymbolChr: str = 'A'
    priorSessionSymbolChrs: list[str] = field(default_factory=list)
    rekeyPrior: bool = True
    autoRename: bool = True
        
    # Decision params
    policyType: DecisionType = DecisionType.FITNESS

    # Current session params
    dataset: str = "nc"
    mission: str = "quad-easy"
    dataDir: Path = Path("/home/benb/data_sets/NewerCollege/ros2")
    bagFile: Path | None = dataDir / mission
    gtFile: Path | None = dataDir / f"ground_truth_tum/gt-nc-{mission}.csv"

    # Save session params
    moduleDir: Path = Path(__file__).parents[1]
    loadSessionDir: Path |None = None
    saveSessionDir: Path = moduleDir / "results/nc/default_session"

    kissICPConfigFile: Path = Path("/home/benb/blocklio/python/params/kiss_icp.yaml")
    
    # Other dataclasses (other structs)
    icp: ICP = field(default_factory=ICP)
    loopClosure: LoopClosure = field(default_factory=LoopClosure)

    ##################################################################
    # Decision Policy Params
    ##################################################################

    # Other dataclasses (other structs)
    distancePolicy: DistancePolicy = field(default_factory=DistancePolicy)
    overlapPolicy: OverlapPolicy = field(default_factory=OverlapPolicy)
    fitnessPolicy: FitnessPolicy = field(default_factory=FitnessPolicy)
    inlierRMSEPolicy: InlierRMSEPolicy = field(default_factory=InlierRMSEPolicy)
    twoStepConnectivityPolicy: TwoStepConnectivityPolicy = field(default_factory=TwoStepConnectivityPolicy)
    relativeCovariancePolicy: RelativeCovariancePolicy = field(default_factory=RelativeCovariancePolicy)
