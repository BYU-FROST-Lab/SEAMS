import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import gtsam
from common.customTypes import TreeData, DecisionType, NodeDecision
from common.weighted_pose_graph_class import WeightedPoseGraph
from common.FGVizHelpers import cleanFactorGraph, getRelativeCov
from params import Params
from copy import copy, deepcopy

class Metrics:
    def __init__(self, method: DecisionType|list[DecisionType], params:Params=Params()):
        if isinstance(method, DecisionType):
            self.method_ = [method]
        else:
            self.method_ = method

        self.params = params

        self.closestDistance = np.inf
        self.lastInterFitness = np.inf


    def CalculateMetrics(self, 
                         isam: gtsam.ISAM2,
                         values: gtsam.Values,
                         priorTreeData: TreeData, 
                         lastPoseKey: int,
                         lastInterICPResult: o3d.pipelines.registration.RegistrationResult|None,
                         pcds: dict[int, o3d.geometry.PointCloud],
                         currentState: NodeDecision,
                         currentKeys: list[int],
                         ):
        
        self.currentState = currentState
        self.lastPoseKey = lastPoseKey

        if DecisionType.DISTANCE in self.method_:
            if priorTreeData is not None:
                self.GetClosestKey(priorTreeData, values.atPose3(lastPoseKey))
            else:
                self.closestDistance = np.inf

        if DecisionType.FITNESS in self.method_:
            if lastInterICPResult is not None:
                self.lastInterFitness = lastInterICPResult.fitness
            else:
                self.lastInterFitness = np.inf
            if self.params.FitnessPolicy.debug:
                print(f"Metrics: FitnessScore = {self.lastInterFitness}")

        if DecisionType.TWO_STEP_VALIDATION in self.method_:
            self.gtsamGraph = cleanFactorGraph(isam)
            self.gtsamValues = isam.calculateEstimate()
            keysToUse = priorTreeData.keys.copy()
            if len(currentKeys) > self.params.fixedLagKeepFrames:
                keysToUse += currentKeys[-self.params.fixedLagKeepFrames:]
            else:
                keysToUse += currentKeys
            self.CreateNXGraph(keysToUse=keysToUse)
            self.ComputeL()

        if DecisionType.RELATIVE_COVARIANCE in self.method_:
            key = self.GetClosestKey(priorTreeData, values.atPose3(lastPoseKey))
            jointMarginal = isam.jointMarginalCovariance([lastPoseKey, key])
            self.relCov = getRelativeCov(values.atPose3(lastPoseKey), values.atPose3(key), jointMarginal.fullMatrix())


    def GetClosestKey(self, priorTreeData: TreeData, pose: gtsam.Pose3):
        d, i = priorTreeData.tree.query(pose.translation())
        self.closestDistance = d
        self.closestPriorKey = priorTreeData.keys[i]

        return priorTreeData.keys[i]

    def CreateNXGraph(self, keysToUse:list[int]=[]):
        self.wpg = WeightedPoseGraph(self.gtsamGraph, self.gtsamValues, keysToUse=keysToUse)

    def ComputeL(self):
        pass