"""
lago submodule
"""
from __future__ import annotations
import python.multisessionDev.docker.dist.typings.gtsam.gtsam
import typing
__all__: list[str] = ['initialize']
@typing.overload
def initialize(graph: python.multisessionDev.docker.dist.typings.gtsam.gtsam.NonlinearFactorGraph, useOdometricPath: bool = True) -> python.multisessionDev.docker.dist.typings.gtsam.gtsam.Values:
    ...
@typing.overload
def initialize(graph: python.multisessionDev.docker.dist.typings.gtsam.gtsam.NonlinearFactorGraph, initialGuess: python.multisessionDev.docker.dist.typings.gtsam.gtsam.Values) -> python.multisessionDev.docker.dist.typings.gtsam.gtsam.Values:
    ...
