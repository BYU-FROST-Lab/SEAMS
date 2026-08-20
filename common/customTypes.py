from scipy.spatial import KDTree
from dataclasses import dataclass, field
from enum import Enum, auto
import open3d as o3d
import numpy as np


@dataclass
class TreeData:
    tree: KDTree
    keys: list[int]
    points: list[list[float]]

@dataclass
class OverlapData:
    closestPriorSubmap: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    # priorEll: float = 0.0
    priorCenterPoint: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    priorMinPoint: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    priorMaxPoint: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    currScan: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    # currEll: float = 0.0
    currCenterPoint: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    currMinPoint: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    currMaxPoint: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

class DecisionType(Enum):
    # Don't make any decisions, just localize
    NONE = auto()

    # geometric methods
    DISTANCE = auto()
    OVERLAP = auto()

    # registration
    FITNESS = auto()
    INLIER_RMSE = auto()

    # graph theoretic
    TWO_STEP_VALIDATION = auto()
    # TWO_OPTION = auto()
    # SUPER_NODE_PRIOR = auto()

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