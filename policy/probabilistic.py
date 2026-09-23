from policy.base import NodeDecisionPolicy
from common.customTypes import NodeDecision
from metrics import Metrics
import numpy as np
from params import Params
from common.fgHelpers import getRelativeCov
import gtsam


class RelativeCovariancePolicy(NodeDecisionPolicy):

    def __init__(self, params: Params):
        super().__init__(params)
        self.initialized = False
        
        
    def DetermineDecisionState(self, metrics: Metrics) -> NodeDecision:

        # if self.params.relativeCovariancePolicy.debug:
        #     print(f"LogDet: {metrics.relCovLogDet}")
        if not self.initialized:
            if metrics.relCovLogDet < self.params.relativeCovariancePolicy.logDetThreshold:
                self.initialized = True
            else:
                return NodeDecision.LOCALIZE

        if metrics.relCovLogDet < self.params.relativeCovariancePolicy.logDetThreshold:
            return NodeDecision.LOCALIZE
        else:
            return NodeDecision.MAP
        
        # threshold = self.params.relativeCovariancePolicy.threshold
        # assert len(threshold) == 3, "Threshold must be a list of length 3 for [x, y, z]"
        # det_threshold = np.linalg.det(np.diag(threshold))
        # det_rel = np.linalg.det(metrics.relCov[3:,3:])
        # relVar = np.diag(metrics.relCov)

        # if self.params.relativeCovariancePolicy.debug:
        #     print(f"Relative covariance between keys \"{gtsam.Symbol(metrics.lastPoseKey).string()}\" and \"{gtsam.Symbol(metrics.closestPriorKey).string()}\"")
        #     print(f"  x: {relVar[3]} y: {relVar[4]} z: {relVar[5]} det: {det_rel}")

        # if relVar[3] > threshold[0] or relVar[4] > threshold[1] or relVar[5] > threshold[2] or det_rel > det_threshold:
        #     if self.params.relativeCovariancePolicy.debug:
        #         print(f"Relative covariance too big between keys \"{gtsam.Symbol(metrics.lastPoseKey).string()}\" and \"{gtsam.Symbol(metrics.closestPriorKey).string()}\"")
        #         print(f"relVal > Threshold")
        #         print(f"x: {relVar[3]} > {threshold[0]} -> {relVar[3] > threshold[0]}")
        #         print(f"y: {relVar[4]} > {threshold[1]} -> {relVar[4] > threshold[1]}")
        #         print(f"z: {relVar[5]} > {threshold[2]} -> {relVar[5] > threshold[2]}")
        #         print(f"det: {det_rel} > {det_threshold} -> {det_rel   > det_threshold}")
        #     if self.initialized:
        #         return NodeDecision.MAP
        #     else:
        #         return NodeDecision.INITIALIZE
        # else:
        #     if not self.initialized:
        #         self.initialized = True
        #     return NodeDecision.LOCALIZE   