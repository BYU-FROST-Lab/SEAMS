from scipy.spatial import KDTree
from dataclasses import dataclass, field

@dataclass
class TreeData:
    tree: KDTree
    keys: list[int]
    points: list[list[float]]