from policy.base import NodeDecisionPolicy
from metrics import Metrics
import numpy as np
from params import Params
from common.customTypes import NodeDecision

class DistancePolicy(NodeDecisionPolicy):
    def DetermineDecisionState(self, metrics: Metrics) -> NodeDecision:
        if not self.initialized:
            if metrics.closestDistance < self.params.distancePolicy.minDistance:
                self.initialized = True
            else:
                return NodeDecision.INITIALIZE
        if np.isfinite(metrics.closestDistance):
            if metrics.closestDistance > self.params.distancePolicy.minDistance:
                return NodeDecision.MAP

        return NodeDecision.LOCALIZE


class OverlapPolicy(NodeDecisionPolicy):
    def DetermineDecisionState(self, metrics: Metrics) -> NodeDecision:
        if not self.initialized:
            if metrics.overlapRatio > self.params.overlapPolicy.overlapThreshold:
                self.initialized = True
            else:
                return NodeDecision.INITIALIZE
        if metrics.overlapRatio < self.params.overlapPolicy.overlapThreshold:
            return NodeDecision.MAP

        return NodeDecision.LOCALIZE
    
    
