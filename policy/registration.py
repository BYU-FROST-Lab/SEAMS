from policy.base import NodeDecisionPolicy
from metrics import Metrics
import numpy as np
from params import Params
from common.customTypes import NodeDecision

class FitnessPolicy(NodeDecisionPolicy):
    def __init__(self, params: Params):
        super().__init__(params)
        if params.FitnessPolicy.fitnessThreshold > 1.:
            raise ValueError(f"FitnessPolicy: fitnessThreshold must be in range [0, 1] and be a float. It's currently {params.FitnessPolicy.fitnessThreshold}")
        
        self.fitnessThreshold_ = params.FitnessPolicy.fitnessThreshold
        self.fitnessHysteresis_ = params.FitnessPolicy.fitnessHysteresisMargin
        self.decision_ = NodeDecision.INITIALIZE

    def DetermineDecisionState(self, metrics: Metrics) -> NodeDecision:
        if not self.initialized:
            if metrics.lastInterFitness > (self.fitnessThreshold_ + self.fitnessHysteresis_):
                self.initialized = True
            else:
                return NodeDecision.INITIALIZE
        if metrics.lastInterFitness < (self.fitnessThreshold_ - self.fitnessHysteresis_):
            self.decision_ = NodeDecision.MAP
        elif metrics.lastInterFitness > (self.fitnessThreshold_ + self.fitnessHysteresis_):
            self.decision_ = NodeDecision.LOCALIZE
        
        return self.decision_
    