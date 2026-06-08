from policy.base import NodeDecisionPolicy
from common.customTypes import NodeDecision
from metrics import Metrics
import numpy as np
from params import Params
from common.FGVizHelpers import getRelativeCov
import gtsam


class RelativeCovariancePolicy(NodeDecisionPolicy):
    def DetermineDecisionState(self, metrics: Metrics) -> NodeDecision:
        
        threshold = self.params.relativeCovariancePolicy.threshold
        det_rel = np.linalg.det(metrics.relCov[:3,:3])
        relVar = np.diag(metrics.relCov)

        if self.params.relativeCovariancePolicy.debug:
            print(f"Relative covariance between keys \"{gtsam.Symbol(metrics.lastPoseKey).string()}\" and \"{gtsam.Symbol(metrics.closestPriorKey).string()}\"")
            print(f"  x: {relVar[0]} y: {relVar[1]} z: {relVar[2]} det: {det_rel}")

        if relVar[0] > threshold[0] or relVar[1] > threshold[1] or relVar[2] > threshold[2] or det_rel > threshold[3]:
            if self.params.relativeCovariancePolicy.debug:
                print(f"Relative covariance too big between keys \"{gtsam.Symbol(metrics.lastPoseKey).string()}\" and \"{gtsam.Symbol(metrics.closestPriorKey).string()}\"")
                print(f"relVal > Threshold")
                print(f"x: {relVar[0]} > {threshold[0]} -> {relVar[0] > threshold[0]}")
                print(f"y: {relVar[1]} > {threshold[1]} -> {relVar[1] > threshold[1]}")
                print(f"z: {relVar[2]} > {threshold[2]} -> {relVar[2] > threshold[2]}")
                print(f"det: {det_rel} > {threshold[3]} -> {det_rel   > threshold[3]}")
            if self.initialized:
                return NodeDecision.MAP
            else:
                return NodeDecision.INITIALIZE
        else:
            if not self.initialized:
                self.initialized = True
            return NodeDecision.LOCALIZE   