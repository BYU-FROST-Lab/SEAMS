from dataclasses import dataclass, field
from base import DecisionType

@dataclass
class Params:
    # Decision params
    decisionMethod: DecisionType = DecisionType.DISTANCE

    # Distance policy
    minDistance: float | int = 20

    # Fitness policy
    fitnessThreshold: float = 0.35
    fitnessHysteresisMargin: float = 0.05

    # ICP params
    icpVoxelSize: float = 0.5
    icpFitnessThreshold: float = 0.3
    icpNearKeyFrames: int = 5

    # Loop Closure params
    loopNoiseScale: float = 1
    loopPrevFrameSkip: int = 100
    loopDistMax: int = 50
    loopSkipFrames: int = 0
    debugLoop: bool = False

    # Fixed Lag Smoother Params
    fixedLagCurrentSession: bool = False
    fixedLagKeepFrames: int = 10

    # Symbol values
    currSessionSymbolChr: str = 'c'
    addToPriorSymbolChr: str = 't'
    priorSessionSymbolChrs: list[str] = field(default_factory=list)

    # Other
    rekeyPrior: bool = True


