import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import gtsam
from common.customTypes import TreeData, DecisionType, NodeDecision, ClosestSubmapData
from common.weighted_pose_graph_class import WeightedPoseGraph
from common.FGVizHelpers import cleanFactorGraph, getRelativeCov, ScanMatch
from params import Params
from copy import copy, deepcopy
import time

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
        self.d_bar = np.inf
        self.lambda2 = np.nan

        # Persistent WeightedPoseGraph for incremental updates (TWO_STEP_VALIDATION)
        self.wpg: WeightedPoseGraph | None = None


    def CalculateMetrics(self, 
                         isam: gtsam.ISAM2,
                         values: gtsam.Values,
                         priorTreeData: TreeData, 
                         lastPoseKey: int,
                         lastInterICPResult: o3d.pipelines.registration.RegistrationResult|None,
                         pcds: dict[int, o3d.geometry.PointCloud],
                         currentState: NodeDecision,
                         currentKeys: list[int],
                         closestSubmapData: ClosestSubmapData,
                         ):
        startTime = time.time()
        
        self.currentState = currentState
        self.lastPoseKey = lastPoseKey

        if DecisionType.DISTANCE in self.method_:
            if priorTreeData is not None:
                self.GetClosestKey(priorTreeData, values.atPose3(lastPoseKey))
            else:
                self.closestDistance = np.inf

        if DecisionType.OVERLAP in self.method_:
            self.ComputeOverlapRatio(closestSubmapData)

        if DecisionType.INLIER_RATIO in self.method_:
            if lastInterICPResult is not None:
                self.lastInterFitness = lastInterICPResult.fitness
            # else:
            #     self.lastInterFitness = 0
            elif closestSubmapData.currKey == lastPoseKey and closestSubmapData.currScan.has_points() and closestSubmapData.closestSubmap.has_points():
                reg = ScanMatch(closestSubmapData.currScan, closestSubmapData.closestSubmap, voxelSize=self.params.loopClosure.voxelSize)
                self.lastInterFitness = reg.fitness
            else:
                self.lastInterFitness = 0

            if self.params.inlierRatioPolicy.debug:
                print(f"Metrics: FitnessScore = {self.lastInterFitness}")

        if DecisionType.INLIER_RMSE in self.method_:
            num_correspondence_set = 0
            if lastInterICPResult is not None:
                self.lastInterInlierRMSE = lastInterICPResult.inlier_rmse if len(lastInterICPResult.correspondence_set) > self.params.inlierRMSEPolicy.minCorrespondenceSet else np.inf
                num_correspondence_set = len(lastInterICPResult.correspondence_set)
            # else:
            #     self.lastInterInlierRMSE = np.inf
            elif closestSubmapData.currKey == lastPoseKey and closestSubmapData.currScan.has_points() and closestSubmapData.closestSubmap.has_points():
                showResult = 2 if self.params.inlierRMSEPolicy.vizScanMatch else 0
                reg = ScanMatch(closestSubmapData.currScan, closestSubmapData.closestSubmap, voxelSize=self.params.loopClosure.voxelSize, showResult=showResult)
                self.lastInterInlierRMSE = reg.inlier_rmse if len(reg.correspondence_set) > self.params.inlierRMSEPolicy.minCorrespondenceSet else np.inf
                num_correspondence_set = len(reg.correspondence_set)
            else:
                self.lastInterInlierRMSE = np.inf

            if np.isclose(self.lastInterInlierRMSE, 0.0):
                self.lastInterInlierRMSE = np.inf
            if self.params.inlierRMSEPolicy.debug:
                print(f"Metrics: InlierRMSE = {self.lastInterInlierRMSE}, Len Correspondences = {num_correspondence_set}")

        if DecisionType.TWO_STEP_VALIDATION in self.method_ or DecisionType.TWO_OPTION in self.method_:
            self.gtsamGraph = cleanFactorGraph(isam)
            self.gtsamValues = isam.calculateEstimate()
            # Build/sync the graph incrementally over prior map + sliding window.
            # currentKeys already contains prior keys followed by the recent window
            # keys as assembled by SessionManager.UpdateDecisionState.
            self.CreateNXGraph(keysToUse=currentKeys)

            startTime = time.time()

            self.ComputeDBar()

        if DecisionType.RELATIVE_COVARIANCE in self.method_:
            # Get the relative covariance between the last pose (next one to be marginalized out) and the closest prior pose
            # if currentState == NodeDecision.LOCALIZE:
            #     assert currentKeys[-1] == lastPoseKey, "current key doesn't equal last pose key"
            #     if not currentKeys:
            #         raise ValueError("Current keys list is empty, cannot compute relative covariance.")
            #     key = self.GetClosestKey(priorTreeData, values.atPose3(currentKeys[-1]))
            #     jointmarginal = isam.jointMarginalCovariance([currentKeys[-1], key])
            #     self.relCov = getRelativeCov(values.atPose3(currentKeys[-1]), values.atPose3(key), jointmarginal.fullMatrix())
            #     self.relCovLogDet = np.log(np.linalg.det(self.relCov))
            # Get the relative covariance between the first pose in the current session and the closest prior pose
            # else:
            key = self.GetClosestKey(priorTreeData, values.atPose3(lastPoseKey))
            jointMarginal = isam.jointMarginalCovariance([lastPoseKey, key])
            self.relCov = getRelativeCov(values.atPose3(lastPoseKey), values.atPose3(key), jointMarginal.fullMatrix())
            self.relCovLogDet = np.log(np.linalg.det(self.relCov))
            if self.params.relativeCovariancePolicy.debug:
                    print(f"Log Det: {self.relCovLogDet:.4f} between keys {gtsam.Symbol(lastPoseKey).string()} and {gtsam.Symbol(key).string()}")

        return time.time() - startTime

    def GetClosestKey(self, priorTreeData: TreeData, pose: gtsam.Pose3):
        d, i = priorTreeData.tree.query(pose.translation())
        self.closestDistance = d
        self.closestPriorKey = priorTreeData.keys[i]

        return priorTreeData.keys[i]

    def CreateNXGraph(self, keysToUse:list[int]=[]):
        """Incrementally sync self.wpg to cover exactly the supplied key list.

        On the first call a new WeightedPoseGraph is created via update().
        Subsequent calls reuse the existing graph, removing stale nodes and
        adding new ones so Laplacian computations stay efficient.
        """
        if self.wpg is None:
            self.wpg = WeightedPoseGraph(criteria=['t_opt', 'e_opt'])
        self.wpg.update(self.gtsamGraph, self.gtsamValues, keysToUse)
        # self.wpg.plot_graph(draw_LC_complex=True)

    def ComputeDBar(self) -> float:
        """Weighted average node degree d-bar from the T-weighted Laplacian.

        d-bar := (1/m) sum_{k=2}^{m} lambda_k = tr(L_T)/m (arXiv:2602.17226, eq. 3).
        """
        self.d_bar = self.wpg.compute_average_degree(weight='weight_t')
        return self.d_bar

    def ComputeFiedler(self) -> float:
        """Weighted Fiedler value lambda_2 of the E-weighted Laplacian."""
        self.lambda2 = self.wpg.compute_fiedler_value(weight='weight_e')
        return self.lambda2

    def ComputeOverlapRatio(self, closestSubmapData: ClosestSubmapData):
        if not closestSubmapData.closestSubmap.has_points() or not closestSubmapData.currScan.has_points():
            self.overlapRatio = 0.0
            return
        prior_xy = np.asarray(closestSubmapData.closestSubmap.points)[:,:2]
        low = 1
        high = 99
        x_lo, x_hi = np.percentile(prior_xy[:, 0], [low, high])
        y_lo, y_hi = np.percentile(prior_xy[:, 1], [low, high])
        submap_min = np.array([x_lo, y_lo])
        submap_max = np.array([x_hi, y_hi])

        current_xy = np.asarray(closestSubmapData.currScan.points)[:,:2]
        x_lo, x_hi = np.percentile(current_xy[:, 0], [low, high])
        y_lo, y_hi = np.percentile(current_xy[:, 1], [low, high])
        current_min = np.array([x_lo, y_lo])
        current_max = np.array([x_hi, y_hi])

        # closestSubmapData.priorMinPoint = submap_min
        # closestSubmapData.priorMaxPoint = submap_max
        # closestSubmapData.currMinPoint = current_min
        # closestSubmapData.currMaxPoint = current_max

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
            tmpPoints = np.asarray(closestSubmapData.closestSubmap.points)
            tmpPoints[:, 2] = 0.0  # Flatten the prior submap
            closestSubmapData.closestSubmap.points = o3d.utility.Vector3dVector(tmpPoints)
            closestSubmapData.closestSubmap.paint_uniform_color(prior_color)
            tmpPoints = np.asarray(closestSubmapData.currScan.points)
            tmpPoints[:, 2] = 0.0  # Flatten the current scan to the XY plane for visualization
            closestSubmapData.currScan.points = o3d.utility.Vector3dVector(tmpPoints)
            closestSubmapData.currScan.paint_uniform_color(current_color)
            # o3d.visualization.draw_geometries([prior_rect, current_rect, closestSubmapData.closestPriorSubmap, closestSubmapData.currScan], window_name="Overlap Visualization", width=800, height=600)

            app = o3d.visualization.gui.Application.instance
            app.initialize()
            vis = o3d.visualization.O3DVisualizer("Overlap Visualization", 800, 600)

            pcd_material = o3d.visualization.rendering.MaterialRecord()
            pcd_material.shader = "defaultUnlit"

            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "unlitLine"
            material.line_width = 3.0

            vis.add_geometry("Prior Submap", closestSubmapData.closestSubmap, pcd_material)
            vis.add_geometry("Current Scan", closestSubmapData.currScan, pcd_material)
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

    def CalcClosestSubmapData(self):

        pcSource = self.FindRecentKeyFrames(0)
        self.closestSubmapData_.currScan = o3d.geometry.PointCloud(pcSource)
        self.closestSubmapData_.currCenterPoint = self.currentEstimate_.atPose3(self.lastOdomKey_).translation()
        self.closestSubmapData_.currKey = copy(self.lastOdomKey_)
        
        foundLoop = False
        currKey = self.lastOdomKey_

        pcSource = self.FindRecentKeyFrames(numNearSourceKeyFrames)
        # pcSource = self.FindRecentKeyFrames(1)
        pcTarget = self.FindNearKeyFrames(key, numNearTargetKeyFrames)

        if self.currentEstimate_.exists(key):
            poseTarget = self.currentEstimate_.atPose3(key)
        elif self.priorValues_.exists(key):
            poseTarget = self.priorValues_.atPose3(key)
        else:
            raise Exception(f"Key {gtsam.Symbol(key).string()} is not found in currentEstimate of priorValues")

        if setSubmap:
            self.closestSubmapData_.closestSubmap = pcTarget
            self.closestSubmapData_.submapCenterPoint = poseTarget.translation()
            self.closestSubmapData_.submapKey = copy(key)