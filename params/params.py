from dataclasses import dataclass, field
from common.customTypes import DecisionType, FixedLagMarginalizationType
from pathlib import Path

# ICP params
@dataclass
class ICP:
    voxelSize: float | None = 0.3   # Either > 0.0 or None. If None, no downsampling is performed.

# Loop Closure params
@dataclass
class LoopClosure:
    noiseScale: float = 1
    prevFrameSkip: int = 100
    distMax: int = 50
    skipFrames: int = 0
    intraNearKeyFrames: int = 5     # Same session
    interNearKeyFrames: int = 5     # Between session. If prior poses have been sparsified (blocks) number should be 0 or small. Otherwise, 5 tends to be sufficient.
    maxPossibleConnections: int = 1
    fitnessThreshold: float = 0.5
    showScanMatchResult: bool = False
    debug: bool = False

# Geometry policy
@dataclass
class DistancePolicy:
    minDistance: float | int = 20

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
    movingAverageLength: int = 40
    showDLplot: bool = True

# Probabilistic
@dataclass
class RelativeCovariancePolicy:
    threshold: list[float] = field(
        default_factory=lambda: [1., 1., 1.]    # [x, y, z, det]
        )
    debug: bool = True


@dataclass
class Params:
    ##################################################################
    # General Multi-Session Params
    ##################################################################

    # Fixed Lag Smoother Params
    fixedLagCurrentSession: bool = False
    fixedLagKeepFrames: int = 50
    fixedLagMarg: FixedLagMarginalizationType = FixedLagMarginalizationType.DENSE

    # Symbol values
    currSessionSymbolChr: str = 'c'
    addToPriorSymbolChr: str = 't'
    priorSessionSymbolChrs: list[str] = field(default_factory=list)
    rekeyPrior: bool = True
    autoRename: bool = True

    # Save session params
    moduleDir: Path = Path(__file__).parents[1]
    loadSessionDir: str | Path |None = None
    saveSessionDir: str | Path = moduleDir / "results/nc/default_session"

    # Current session params
    dataDir: Path = Path("/home/benb/data_sets/NewerCollege/ros2")
    bagFile: Path | None = dataDir / "quad-easy"
    gtFile: Path | None = dataDir / "ground_truth_tum/gt-nc-quad-hard.csv"

    kissICPConfigFile: Path = Path("/home/benb/blocklio/python/params/kiss_icp.yaml")
    
    # Other dataclasses (other structs)
    icp: ICP = field(default_factory=ICP)
    loopClosure: LoopClosure = field(default_factory=LoopClosure)

    ##################################################################
    # Decision Policy Params
    ##################################################################
        
    # Decision params
    policyType: DecisionType = DecisionType.DISTANCE

    # Other dataclasses (other structs)
    distancePolicy: DistancePolicy = field(default_factory=DistancePolicy)
    fitnessPolicy: FitnessPolicy = field(default_factory=FitnessPolicy)
    inlierRMSEPolicy: InlierRMSEPolicy = field(default_factory=InlierRMSEPolicy)
    twoStepConnectivityPolicy: TwoStepConnectivityPolicy = field(default_factory=TwoStepConnectivityPolicy)
    relativeCovariancePolicy: RelativeCovariancePolicy = field(default_factory=RelativeCovariancePolicy)
