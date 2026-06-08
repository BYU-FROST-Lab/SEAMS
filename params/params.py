from dataclasses import dataclass, field
from common.customTypes import DecisionType

@dataclass
class Params:
    ##################################################################
    # General Multi-Session Params
    ##################################################################

    # Fixed Lag Smoother Params
    fixedLagCurrentSession: bool = False
    fixedLagKeepFrames: int = 10

    # Symbol values
    currSessionSymbolChr: str = 'c'
    addToPriorSymbolChr: str = 't'
    priorSessionSymbolChrs: list[str] = field(default_factory=list)
    rekeyPrior: bool = True

    # ICP params
    @dataclass
    class ICP:
        voxelSize: float = 0.5
        fitnessThreshold: float = 0.3
        nearKeyFrames: int = 5

    # Loop Closure params
    @dataclass
    class LoopClosure:
        noiseScale: float = 1
        prevFrameSkip: int = 30
        distMax: int = 30
        skipFrames: int = 0
        maxPossibleConnections: int = 1
        debug: bool = False

    ##################################################################
    # Decision Policy Params
    ##################################################################
        
    # Decision params
    policyType: DecisionType = DecisionType.DISTANCE

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

    # Graph Theoretic
    @dataclass
    class TwoStepConnectivityPolicy:
        movingAverageLength: int = 40
        showDLplot: bool = True

    # Probabilistic
    @dataclass
    class RelativeCovariancePolicy:
        threshold: list[float] = field(
            default_factory=lambda: [0.01, 0.01, 0.01, 0.001]    # [x, y, z, det]
            )
        debug: bool = True
        
    relativeCovariancePolicy: RelativeCovariancePolicy = field(
        default_factory=RelativeCovariancePolicy
    )




