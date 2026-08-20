import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import gtsam
from common.customTypes import TreeData, DecisionType, NodeDecision, OverlapData
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
        self.overlapRatio = 0.0


    def CalculateMetrics(self, 
                         isam: gtsam.ISAM2,
                         values: gtsam.Values,
                         priorTreeData: TreeData, 
                         lastPoseKey: int,
                         lastInterICPResult: o3d.pipelines.registration.RegistrationResult|None,
                         pcds: dict[int, o3d.geometry.PointCloud],
                         currentState: NodeDecision,
                         currentKeys: list[int],
                         overlapData: OverlapData,
                         ):
        
        self.currentState = currentState
        self.lastPoseKey = lastPoseKey

        if DecisionType.DISTANCE in self.method_:
            if priorTreeData is not None:
                self.GetClosestKey(priorTreeData, values.atPose3(lastPoseKey))
            else:
                self.closestDistance = np.inf

        if DecisionType.OVERLAP in self.method_:
            self.ComputeOverlapRatio(overlapData)

        if DecisionType.FITNESS in self.method_:
            if lastInterICPResult is not None:
                self.lastInterFitness = lastInterICPResult.fitness
            else:
                self.lastInterFitness = np.inf
            if self.params.fitnessPolicy.debug:
                print(f"Metrics: FitnessScore = {self.lastInterFitness}")

        if DecisionType.INLIER_RMSE in self.method_:
            if lastInterICPResult is not None:
                self.lastInterInlierRMSE = lastInterICPResult.inlier_rmse
            else:
                self.lastInterInlierRMSE = np.inf
            if self.params.inlierRMSEPolicy.debug:
                print(f"Metrics: InlierRMSE = {self.lastInterInlierRMSE}")

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
            # Get the relative covariance between the last pose (next one to be marginalized out) and the closest prior pose
            if currentState == NodeDecision.LOCALIZE:
                if not currentKeys:
                    raise ValueError("Current keys list is empty, cannot compute relative covariance.")
                key = self.GetClosestKey(priorTreeData, values.atPose3(currentKeys[-1]))
                jointmarginal = isam.jointMarginalCovariance([currentKeys[-1], key])
                self.relCov = getRelativeCov(values.atPose3(currentKeys[-1]), values.atPose3(key), jointmarginal.fullMatrix())
            # Get the relative covariance between the first pose in the current session and the closest prior pose
            else:
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

    def ComputeOverlapRatio(self, overlapData: OverlapData):
        prior_xy = np.asarray(overlapData.closestPriorSubmap.points)[:,:2]
        low = 1
        high = 99
        x_lo, x_hi = np.percentile(prior_xy[:, 0], [low, high])
        y_lo, y_hi = np.percentile(prior_xy[:, 1], [low, high])
        submap_min = np.array([x_lo, y_lo])
        submap_max = np.array([x_hi, y_hi])

        current_xy = np.asarray(overlapData.currScan.points)[:,:2]
        x_lo, x_hi = np.percentile(current_xy[:, 0], [low, high])
        y_lo, y_hi = np.percentile(current_xy[:, 1], [low, high])
        current_min = np.array([x_lo, y_lo])
        current_max = np.array([x_hi, y_hi])

        # overlapData.priorMinPoint = submap_min
        # overlapData.priorMaxPoint = submap_max
        # overlapData.currMinPoint = current_min
        # overlapData.currMaxPoint = current_max

        intersection_min = np.maximum(current_min, submap_min)
        intersection_max = np.minimum(current_max, submap_max)

        intersection_size = np.maximum(0.0, intersection_max - intersection_min)
        intersection_area = np.prod(intersection_size)

        square_area = np.prod(current_max - current_min)
        if square_area == 0:
            raise ValueError("Current scan bounding box has zero area, cannot compute overlap ratio.")
        self.overlapRatio = intersection_area / square_area

        if self.params.overlapPolicy.debug:
            prior_color = (1.0, 0.0, 0.0)  # Red
            current_color = (0.0, 1.0, 0.0)
            prior_rect = self.createO3dRect(submap_min, submap_max, (0.0, 0.0, 1.0))  # Blue
            current_rect = self.createO3dRect(current_min, current_max, (0.0, 1.0, 1.0))  # Cyan

            # color point clouds for visualization
            tmpPoints = np.asarray(overlapData.closestPriorSubmap.points)
            tmpPoints[:, 2] = 0.0  # Flatten the prior submap
            overlapData.closestPriorSubmap.points = o3d.utility.Vector3dVector(tmpPoints)
            overlapData.closestPriorSubmap.paint_uniform_color(prior_color)
            tmpPoints = np.asarray(overlapData.currScan.points)
            tmpPoints[:, 2] = 0.0  # Flatten the current scan to the XY plane for visualization
            overlapData.currScan.points = o3d.utility.Vector3dVector(tmpPoints)
            overlapData.currScan.paint_uniform_color(current_color)
            # o3d.visualization.draw_geometries([prior_rect, current_rect, overlapData.closestPriorSubmap, overlapData.currScan], window_name="Overlap Visualization", width=800, height=600)

            app = o3d.visualization.gui.Application.instance
            app.initialize()
            vis = o3d.visualization.O3DVisualizer("Overlap Visualization", 800, 600)

            pcd_material = o3d.visualization.rendering.MaterialRecord()
            pcd_material.shader = "defaultUnlit"

            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "unlitLine"
            material.line_width = 3.0

            vis.add_geometry("Prior Submap", overlapData.closestPriorSubmap, pcd_material)
            vis.add_geometry("Current Scan", overlapData.currScan, pcd_material)
            vis.add_geometry("Prior Rect", prior_rect, material)
            vis.add_geometry("Current Rect", current_rect, material)

            # Make sure everything is framed.
            vis.reset_camera_to_default()

            app.add_window(vis)
            app.run()
        pass

    def createO3dRect(self, min_point: np.ndarray, max_point: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
        """
        Create a 2D rectangle in the XY plane using Open3D LineSet.
        """
        if min_point.shape != (2,) or max_point.shape != (2,):
            raise ValueError("min_point and max_point must be 2D points.")

        # Define the corners of the rectangle
        corners = np.array([
            [min_point[0], min_point[1], 10],
            [max_point[0], min_point[1], 10],
            [max_point[0], max_point[1], 10],
            [min_point[0], max_point[1], 10]
        ])

        # Define the lines connecting the corners
        lines = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0]
        ])

        # Create a LineSet object
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

        return line_set