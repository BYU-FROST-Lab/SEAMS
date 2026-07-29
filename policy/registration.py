from policy.base import NodeDecisionPolicy
from metrics import Metrics
import numpy as np
from params import Params
from common.customTypes import NodeDecision

class FitnessPolicy(NodeDecisionPolicy):
    def __init__(self, params: Params):
        super().__init__(params)
        if params.fitnessPolicy.fitnessThreshold > 1.:
            raise ValueError(f"FitnessPolicy: fitnessThreshold must be in range [0, 1] and be a float. It's currently {params.fitnessPolicy.fitnessThreshold}")
        
        self.fitnessThreshold_ = params.fitnessPolicy.fitnessThreshold
        self.fitnessHysteresis_ = params.fitnessPolicy.fitnessHysteresisMargin
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
    
class InlierRMSEPolicy(NodeDecisionPolicy):
    def __init__(self, params: Params):
        super().__init__(params)
        if params.inlierRMSEPolicy.inlierRMSEThreshold < 0.:
            raise ValueError(f"InlierRMSEPolicy: inlierRMSEThreshold must be >= 0 and be a float. It's currently {params.inlierRMSEPolicy.inlierRMSEThreshold}")
        
        self.inlierRMSEThreshold_ = params.inlierRMSEPolicy.inlierRMSEThreshold
        self.inlierRMSEHysteresis_ = params.inlierRMSEPolicy.inlierRMSEHysteresisMargin
        self.decision_ = NodeDecision.INITIALIZE

    def DetermineDecisionState(self, metrics: Metrics) -> NodeDecision:
        if not self.initialized:
            if metrics.lastInterInlierRMSE < (self.inlierRMSEThreshold_ - self.inlierRMSEHysteresis_):
                self.initialized = True
            else:
                return NodeDecision.INITIALIZE
        if metrics.lastInterInlierRMSE > (self.inlierRMSEThreshold_ + self.inlierRMSEHysteresis_):
            self.decision_ = NodeDecision.MAP
        elif metrics.lastInterInlierRMSE < (self.inlierRMSEThreshold_ - self.inlierRMSEHysteresis_):
            self.decision_ = NodeDecision.LOCALIZE
        
        return self.decision_