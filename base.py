from abc import ABC, abstractmethod
from enum import Enum, auto

class DecisionType(Enum):
    # naive methods
    DISTANCE = auto()
    FITNESS = auto()
    OVERLAP = auto()

class NodeDecision(Enum):
    LOCALIZE = auto()
    MAP = auto()

class NodeDecisionPolicy(ABC):

    @abstractmethod
    def DetermineDecisionState(self, metrics) -> NodeDecision:
        pass