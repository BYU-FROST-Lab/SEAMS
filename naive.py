from base import NodeDecisionPolicy, NodeDecision
from metrics import Metrics
import numpy as np
from params import Params

class DistancePolicy(NodeDecisionPolicy):
    def __init__(self, params: Params):
        self.min_distance_ = params.minDistance

    def DetermineDecisionState(self, metrics: Metrics) -> NodeDecision:
        if np.isfinite(metrics.closestDistance):
            if metrics.closestDistance > self.min_distance_:
                return NodeDecision.MAP

        return NodeDecision.LOCALIZE
    
class FitnessPolicy(NodeDecisionPolicy):
    def __init__(self, params: Params):
        if params.fitnessThreshold > 1.:
            raise ValueError(f"FitnessPolicy: fitnessThreshold must be in range [0, 1] and be a float. It's currently {fitnessThreshold}")
        
        self.fitnessThreshold_ = params.fitnessThreshold
        self.fitnessHysteresis_ = params.fitnessHysteresisMargin
        self.decision_ = NodeDecision.LOCALIZE

    def DetermineDecisionState(self, metrics: Metrics) -> NodeDecision:
        if metrics.lastInterFitness < (self.fitnessThreshold_ - self.fitnessHysteresis_):
            self.decision_ = NodeDecision.MAP
        elif metrics.lastInterFitness > (self.fitnessThreshold_ + self.fitnessHysteresis_):
            self.decision_ = NodeDecision.LOCALIZE
        
        return self.decision_