import numpy as np
import open3d as o3d
from policy.base import DecisionType
from scipy.spatial import KDTree
import gtsam
from common.customStructs import TreeData

class Metrics:
    def __init__(self, method: DecisionType|list[DecisionType]):
        if isinstance(method, DecisionType):
            self.method_ = [method]
        else:
            self.method_ = method

        self.closestDistance = np.inf
        self.lastInterFitness = np.inf


    def CalculateMetrics(self, 
                         isam: gtsam.NonlinearFactorGraph,
                         values: gtsam.Values,
                         treeData: TreeData|None, 
                         lastPoseKey: int,
                         lastInterICPResult: o3d.pipelines.registration.RegistrationResult|None,
                         pcds: dict[int, o3d.geometry.PointCloud],

                         ):
        if DecisionType.DISTANCE in self.method_:
            if treeData is not None:
                self.GetClosestKey(treeData, values.atPose3(lastPoseKey))
            else:
                self.closestDistance = np.inf
        if DecisionType.FITNESS in self.method_:
            # self.CalcFitnessScore(values, lastPoseKey, treeData, pcds)
            if lastInterICPResult is not None:
                self.lastInterFitness = lastInterICPResult.fitness
            else:
                self.lastInterFitness = np.inf
            print(f"Metrics: FitnessScore = {self.lastInterFitness}")


    def GetClosestKey(self, treeData: TreeData, pose: gtsam.Pose3):
        d, i = treeData.tree.query(pose.translation())
        self.closestDistance = d

        return treeData.keys[i]


    # def CalcFitnessScore(self, values, lastPoseKey, treeData, pcds):

    #     pose = values.atPose3(lastPoseKey)
    #     self.UpdatePriorTree()
    #     distances, indices = self.priorTree_.tree.query(pose.translation(), k=5, distance_upper_bound=self.params.loopDistMax) # Look for the 5 closest poses given the distance is less than 30
    #     keys = deepcopy(self.priorTree_.keys)

    #     keysToSearch = np.asarray(keys)[indices[distances < self.params.loopDistMax]]
    #     reg = self.LoopCheckGivenKeys(keysToSearch)
    #     if reg:
    #         self.lastInterICPResult_ = reg