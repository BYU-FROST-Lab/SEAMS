"""
gtsfm submodule
"""
from __future__ import annotations
import python.multisessionDev.docker.dist.typings.gtsam.gtsam
import numpy
__all__: list[str] = ['Keypoints', 'tracksFromPairwiseMatches']
class Keypoints:
    coordinates: numpy.ndarray[numpy.float64[m, 2]]
    def __init__(self, coordinates: numpy.ndarray[numpy.float64[m, 2]]) -> None:
        ...
def tracksFromPairwiseMatches(matches_dict: dict[python.multisessionDev.docker.dist.typings.gtsam.gtsam.IndexPair, numpy.ndarray[numpy.int32[m, 2]]], keypoints_list: list[Keypoints], verbose: bool = False) -> list[python.multisessionDev.docker.dist.typings.gtsam.gtsam.SfmTrack2d]:
    ...
