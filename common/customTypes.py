from scipy.spatial import KDTree
from dataclasses import dataclass, field
from enum import Enum, auto


@dataclass
class TreeData:
    tree: KDTree
    keys: list[int]
    points: list[list[float]]

class DecisionType(Enum):
    # Don't make any decisions, just localize
    NONE = auto()

    # geometric methods
    DISTANCE = auto()

    # registration
    FITNESS = auto()
    INLIER_RMSE = auto()

    # graph theoretic
    TWO_STEP_VALIDATION = auto()
    TWO_OPTION = auto()
    SUPER_NODE_PRIOR = auto()

    # probablistic
    RELATIVE_COVARIANCE = auto()



class NodeDecision(Enum):
    LOCALIZE = auto()
    MAP = auto()
    INITIALIZE = auto()


class FixedLagMarginalizationType(Enum):
    DENSE = auto()
    APPROX_PRIOR = auto()
    UNARY = auto()
    PRIOR_ON_ANCHOR_ALL = auto()
    PRIOR_ON_ANCHOR_FIRST = auto()
    THROW_AWAY_BETWEEN = auto()