#!/usr/bin/env python3

import os
import shutil

import numpy as np

np.set_printoptions(linewidth=np.inf, suppress=True)
from copy import copy, deepcopy
from pathlib import Path
import open3d as o3d
from scipy.spatial import KDTree
import time
from collections import deque
from dataclasses import asdict
import yaml

# Local imports
import common.FGVizHelpers as fgtools
import gtsam
import gtsam_unstable
from params import Params
from common.customTypes import *
from metrics import Metrics
from policy.base import NoDecisionPolicy
from policy.geometric import DistancePolicy, OverlapPolicy
from policy.registration import InlierRatioPolicy, InlierRMSEPolicy
from policy.graphTheoretic import TwoStepConnectivityPolicy
from policy.probabilistic import RelativeCovariancePolicy

class SessionManager:
    class UpdateResult:
        key: int = -1
        numIntraLoops: int = 0
        numInterLoops: int = 0

    def __init__(self, priorSessionDir:str|Path|None=None, params:Params=Params()):
        self.params = params
        
        self.graph_ = gtsam.NonlinearFactorGraph()
        self.values_ = gtsam.Values()

        self.smoother_ = gtsam.IncrementalFixedLagSmoother(float(self.params.fixedLagKeepFrames))
        self.currTimestamp_ = 0.0
        self.currentEstimate_ = gtsam.Values()

        self.pcds_: dict[int, o3d.geometry.PointCloud] = {}
        self.closestSubmapData_: ClosestSubmapData = ClosestSubmapData()
        self.timestamps_: dict[int, float] = {}

        self.loopCnt_ = 0
        self.interLoopValues_ = gtsam.Values()

        if self.params.policyType == DecisionType.NONE:
            self.decider = NoDecisionPolicy(self.params)
        elif self.params.policyType == DecisionType.DISTANCE:
            self.decider = DistancePolicy(self.params)
        elif self.params.policyType == DecisionType.OVERLAP:
            self.decider = OverlapPolicy(self.params)
        elif self.params.policyType == DecisionType.INLIER_RATIO:
            self.decider = InlierRatioPolicy(self.params)
        elif self.params.policyType == DecisionType.INLIER_RMSE:
            self.decider = InlierRMSEPolicy(self.params)
        elif self.params.policyType == DecisionType.TWO_STEP_VALIDATION:
            self.decider = TwoStepConnectivityPolicy(self.params)
        elif self.params.policyType == DecisionType.RELATIVE_COVARIANCE:
            self.decider = RelativeCovariancePolicy(self.params)
        else:
            raise ValueError(f"SessionManager: Invalid decision policy chosen. {self.params.policyType}")
        
        self.metrics_ = Metrics(self.params.policyType)
        self.timeToCalcMetricsAndDecision_ = []
        self.timeToOptimize_ = []

        self.mappingStartKeys_: list[int] = []
        self.mappingEndKeys_: list[int] = []
        self.mappingStartKeyDets_: dict[int, list] = {}
        self.mappingEndKeyDets_: dict[int, list] = {}
        self.relCovStartKeys_: list[int] = []

        self.loopKeysToIgnore_: set[int]  = set()
        self.lastInterICPResult_ = None
        self.recentKeysQueue_: deque[int] = deque(maxlen=max(self.params.loopClosure.prevFrameSkip, self.params.fixedLagKeepFrames) + 1)
        
        self.currentSessionSymbol_ = self.params.currSessionSymbolChr
        self.addToPriorSymbol_ = self.params.addToPriorSymbolChr
        self.priorSessionSymbols_ = self.params.priorSessionSymbolChrs

        self.sessionKeys: dict[str, list[int]] = {}
        self.lastOdomKey_ = 0
        # self.localWindowSubmap: tuple[o3d.geometry.PointCloud, int] = (o3d.geometry.PointCloud(), -1)

        self.initializationCount_ = 0
        self.initialFactorInfo_ = []

        # GPS
        self.refLatitude_ = np.inf
        self.refLongitude_ = np.inf
        self.refAltitude_ = np.inf

        # Prior graph
        self.priorGraph_ = gtsam.NonlinearFactorGraph()
        self.priorValues_ = gtsam.Values()
        self.interLoopTree_: TreeData | None = None

        self.currentToPriorAnchors_: dict[int, list[int]] = {}
        self.priorAnchorsToCurrent_: dict[int, list[int]] = {}
        self.priorKeyToFactorIndexDict_: dict[int, list[int]] = {}

        if priorSessionDir is not None:
            self.LoadSession(priorSessionDir)
        self.decisionState_ = NodeDecision.INITIALIZE

        self.ResetMode()

        # Noise models
        self.odomNoise_ = gtsam.noiseModel.Diagonal.Variances(np.asarray([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]))


    def ResetMode(self, mode: NodeDecision | None = None):
        '''
        Reset variables specific to these modes
        '''

        if mode == NodeDecision.INITIALIZE or mode == None:
            pass
        if mode == NodeDecision.LOCALIZE or mode == None:
            pass
        if mode == NodeDecision.MAP or mode == None:
            self.intraLoopKeys_:list[int] = []
            self.intraLoopDets_:list = []
            self.intraLoopReset_: int = 0
            
            pass

    
    def LoadPriorGraphAndValues(self, filename:str, loadGPS:bool=True, symbolChar:list[str]|None=None) -> None: 
        graph, values = fgtools.readG2oWithPrior(filename, loadGPS=loadGPS)
        self.SetPriorGraphAndValues(graph, values, symbolChar)
        
    def SetPriorGraphAndValues(self, graph: gtsam.gtsam.NonlinearFactorGraph, values:gtsam.gtsam.Values, symbolChar:list[str]|None=None) -> None:

        graph, values = self.RenameKeys(symbolChar, graph, values, autoRename=self.params.autoRename, updateOptimization=False)
        self.priorGraph_ = deepcopy(graph)
        self.priorValues_ = deepcopy(values)
        self.IncrementSize(self.priorValues_.keys())
        self.UpdatePriorTree()
        # self.UpdatePriorKeyToFactorIndexDict()

        # Used when creating dense factor for prior map
        self.UpdateSmoother()

    def UpdateSmoother(self, reset:bool=False, 
                             newLag:float|None=None,
                             optimizationTimeList:list|None=None):
        # Reset if needed
        if reset:
            self.smoother_ = gtsam.IncrementalFixedLagSmoother(float(self.params.fixedLagKeepFrames))
            self.currentEstimate_ = gtsam.Values()
        
        if newLag is not None:
            self.smoother_.setSmootherLag(newLag) # Added in custom wheel

        self.IncrementSize(self.values_.keys())
        timestamps = {}

        # Include full prior Graph in fixed lag
        if self.currentEstimate_.size() == 0:
            for k in self.priorValues_.keys():
                timestamps[k] = self.currTimestamp_
            self.smoother_.update(self.priorGraph_, self.priorValues_, timestamps, [], gtsam.KeySet(self.priorValues_.keys()))

        keysToFreeze = gtsam.KeySet()
        for k in self.values_.keys():
            timestamps[k] = self.currTimestamp_
            if self.IsAddToPriorSessionKey(k):
                keysToFreeze.insert(k)

        startTime = time.time()
        result = self.smoother_.update(self.graph_, self.values_, timestamps, [], keysToFreeze)
        if isinstance(optimizationTimeList, list):
            optimizationTimeList.append(time.time() - startTime)

        self.graph_.resize(0)
        self.values_.clear()
        # try:
        self.currentEstimate_ = self.smoother_.calculateEstimate()
        # except Exception as e:
        #     print(f"Error occurred while calculating estimate: {e}")
        #     pass

        return result

    def ApproximateLinearFactor(self, marginalFactorsIdxs: list, maxNodesToKeep: int=1):
        if maxNodesToKeep < 1:
            raise ValueError(f"maxNodesToKeep must be greater than 0, but got {maxNodesToKeep}")
        if not marginalFactorsIdxs or (not self.currentToPriorAnchors_ and maxNodesToKeep == 1):
            return
        # Find the linear factors
        currGraph = self.smoother_.getFactors()
        for idx in marginalFactorsIdxs:
            f = currGraph.at(idx)

            # Factors can be null due to the way the fixed lag smoother works. If a factor is null, it means it has been marginalized out and we can skip it.
            if not f:
                continue
            
            # Don't need to worry about unary or binary factors.
            if len(f.keys()) <= maxNodesToKeep:
                # self.priorIdx_ = idx
                continue

            if isinstance(f, gtsam.LinearContainerFactor):
                hf = f.toHessian()
                keys = hf.keys()

                aug_info = hf.augmentedInformation()

                G = aug_info[:-1,:-1] # Information matrix portion
                g = aug_info[:-1, -1] # Information vector portion
                c = aug_info[ -1, -1] # Constant term (not used)

                dim = 6 # Pose3
                n = len(keys)
                assert G.size == (dim*n * dim*n), f"Expected hessian to be of size {(dim*n * dim*n)} but is actually {G.size}"

                ####################################################################################
                # Approximation using the Schur complement to marginalize out the prior graph nodes.
                ####################################################################################
                # Determine which keys to keep based on closest distance between the prior and current session keys.

                remove_key_idxs = []
                points, _ = fgtools.GetPointsFromGraphEstimate(self.currentEstimate_, keys=keys)
                points = np.array(points)
                # gtsam keys are sorted. Assuming the convention of prior sessions having capital letters "A, B, etc."
                # and the current session having symbol letter "c" then the last node should always be the most recent
                # in the current session.
                current_point = points[-1]
                distances = [np.linalg.norm(point - current_point) for point in points[:-1]]
                closest_idxs = np.argsort(distances)[:maxNodesToKeep-1]

                for i in range(len(keys)):
                    if i not in closest_idxs and i != len(keys) - 1:
                        remove_key_idxs.append(i)


                # Get indices for the prior and current session keys in the HessianFactor
                remove_idxs = []
                keep_idxs = []
                keep_keys = []
                for i, k in enumerate(keys):
                    indices = list(range(i * dim, (i + 1) * dim))
                    if i in remove_key_idxs:
                        remove_idxs.extend(indices)
                    else:
                        keep_idxs.extend(indices)
                        keep_keys.append(k)

                # Extract Schur Complement blocks
                Grr = G[np.ix_(remove_idxs, remove_idxs)]
                Grk = G[np.ix_(remove_idxs, keep_idxs)]
                Gkk = G[np.ix_(keep_idxs, keep_idxs)]

                gr = g[remove_idxs]
                gk = g[keep_idxs]

                # Compute the Schur complement
                G_prime = Gkk - Grk.T @ np.linalg.inv(Grr) @ Grk
                g_prime = gk - Grk.T @ np.linalg.inv(Grr) @ gr

                new_aug_info = np.zeros((len(keep_idxs) + 1, len(keep_idxs) + 1))
                new_aug_info[:-1, :-1] = G_prime
                new_aug_info[:-1, -1] = g_prime
                new_aug_info[-1, -1] = c

                # Create new binary factor
                graph = gtsam.NonlinearFactorGraph()
                # new_hf = gtsam.HessianFactor(keys[current_key_idxs], G_prime, g_prime, c)
                block_dims = [dim] * len(keep_keys) + [1]
                new_hf = gtsam.HessianFactor(keep_keys, gtsam.SymmetricBlockMatrix(block_dims, new_aug_info))
                graph.add(gtsam.LinearContainerFactor(new_hf))
                assert np.allclose(G_prime, new_hf.information()), f"Expected G_prime to be equal to new_hf.information() but they are not equal. G_prime: {G_prime}, new_hf.information(): {new_hf.information()}"
                
                self.smoother_.update(graph, gtsam.Values(), {}, [idx])

    def UpdateDecisionState(self) -> bool:
        '''
        This function updates the decision state based on the current metrics and the decider policy.
        
        :return: True if the decision state has changed, False otherwise'''
        ret = False
        previousState = copy(self.decisionState_)
        if self.interLoopTree_ is None:
            return False
        metricTime = self.metrics_.CalculateMetrics(
            self.smoother_.getISAM2(), 
            self.currentEstimate_, 
            self.interLoopTree_, 
            self.lastOdomKey_,
            self.lastInterICPResult_,
            self.pcds_,
            self.decisionState_,
            self.sessionKeys[self.currentSessionSymbol_] if self.currentSessionSymbol_ in self.sessionKeys else [],
            self.closestSubmapData_,
            )
        startTime = time.time()
        self.decisionState_ = self.decider.DetermineDecisionState(self.metrics_)
        decisionTime = time.time() - startTime
        self.timeToCalcMetricsAndDecision_.append(metricTime + decisionTime)

        if self.decisionState_ == NodeDecision.INITIALIZE:
            print(f"DecisionState - {self.params.policyType.name}: Waiting for policy initialization...")
        
        if previousState != self.decisionState_:
            print(f"DecisionState - {self.params.policyType.name}: "
                  f"{'Localize' if self.decisionState_ == NodeDecision.LOCALIZE else ('Map' if self.decisionState_ == NodeDecision.MAP else 'Initialize')}")

        if previousState == NodeDecision.LOCALIZE and self.decisionState_ == NodeDecision.MAP:
            ret = True
            # if self.params.policyType == DecisionType.RELATIVE_COVARIANCE:
                # self.RenameKeysInPlace({self.currentSessionSymbol_: self.addToPriorSymbol_})
            pass
        elif previousState == NodeDecision.MAP and self.decisionState_ != NodeDecision.MAP and len(self.recentKeysQueue_) > 1:
            if self.params.priorUpdateMethod > 0:
                self.UpdatePriorTree()
            ret = True

        return ret

    def RenameKeys(self, 
                   chars: str|list[str]|None=None, 
                   graph:gtsam.NonlinearFactorGraph|None=None, 
                   values:gtsam.Values|None=None,
                   autoRename:bool=False,
                   updateOptimization:bool=True, 
                   resetOptimization:bool=False):
        '''
        This function is used to rename the keys in the object graph and values or in the provided graph and values.
        If chars is set to None (default), the keys will automatically be labeled in order from 'A' to 'Z'

        :param chars: Either single character, list of single characters, or None. If None, autoRename will be marked True. 
        :param graph: (Optional) If not provided, will default to the object's graph
        :param values: (Optional) If not provided, will default to the object's values
        :param autoRename: (default=False) If True, the symbol characters will be determined starting from 'A' to 'Z'
        :param updateOptimization: (default=True) automatically update optimization with the graph and values
        :param resetOptimization: (default=False) If updateOptimization is True, this can reset the optimization graph 
            and values and add in only the rekeyed graph and values computed in this function.
        '''

        if graph is None or values is None:
            graph = self.priorGraph_
            values = self.priorValues_
        
        if chars is None or autoRename:
            autoRename = True
            chars = [chr(i) for i in range(ord('A'), ord('A')+26)]
        elif isinstance(chars, str):
            chars = [chars]
            print(f"Renaming keys to start with '{chars}'")
        else:
            for char in chars:
                if len(char) != 1:
                    raise ValueError(f"RenameKeys: 'char' must be a single character. Current char = '{char}'")
            print(f"Renaming keys to start with '{chars}'")
        
        usedChars: dict[int, str] = {}
        iterChar = iter(chars)

        def renameKey(oldKey):
            s = gtsam.Symbol(oldKey)
            if s.chr() not in usedChars:
                try:
                    usedChars[s.chr()] = next(iterChar)
                except StopIteration as e:
                    raise StopIteration("RenameKeys: More symbols are in the session values than rekey symbols provided in the input list")

            return gtsam.Symbol(usedChars[s.chr()], s.index()).key()
        
        keyMap = {}
        for key in values.keys():
            keyMap[key] = renameKey(key)
        
        new_graph = gtsam.NonlinearFactorGraph()
        for i in range(graph.size()):
            f = graph.at(i)

            new_keys = [keyMap[k] if k in keyMap else k for k in f.keys()]
            
            # The list length MUST equal f.keys().size()
            assert len(new_keys) == len(f.keys()), f"Expected {len(f.keys())} keys, got {len(new_keys)}"

            new_f = f.rekey(new_keys)
            new_graph.push_back(new_f)

        new_values = gtsam.Values()
        for old_key, new_key in keyMap.items():
            new_values.insert(new_key, values.atPose3(old_key))


        if self.pcds_:
            pcdKeys = list(self.pcds_.keys())
            for key in pcdKeys:
                if key in keyMap.keys():
                    # if resetOptimization:
                    self.pcds_[keyMap[key]] = self.pcds_.pop(key)
                    self.timestamps_[keyMap[key]] = self.timestamps_.pop(key)
                    # else:
                    #     self.pcds_[keyMap[key]] = self.pcds_[key]
                    #     self.timestamps_[keyMap[key]] = self.timestamps_[key]
            
            if not set(new_values.keys()).issubset(self.pcds_.keys()):
                print(f"pcd keys don't match value's keys{[f" {gtsam.Symbol(k).string()}" for k in (set(new_values.keys()) - set(self.pcds_.keys()))]}")
        
        if autoRename:
            self.priorSessionSymbols_ = list(usedChars.values())
            self.addToPriorSymbol_ = chr(ord(self.priorSessionSymbols_[-1]) + 1)

            print(f"PriorSessionSymbols have been updated to {self.priorSessionSymbols_}")
            print(f"AddToPriorSymbol has been updated to {self.addToPriorSymbol_}")
        
        if updateOptimization:
            self.graph_.push_back(new_graph)
            self.values_.insert(new_values)
            self.UpdateSmoother(reset=resetOptimization)
        
        return new_graph, new_values
    
    def RenameKeysInPlace(self, charMap: dict[str, str], keys: list[int] = [], keepKeyIndex:bool=True, freezeNewNodes:bool=True):
        
        graph = self.smoother_.getFactors()
        timeMap = self.smoother_.timestamps()
        recentKeysQueue = list(self.recentKeysQueue_)
        
        # Create key map and new Values object with rekeyed values
        keyMap: dict[int, int] = {}
        newValues = gtsam.Values()
        if keys:
            keysToCheck = keys
        else:
            keysToCheck = self.currentEstimate_.keys()
        for k in keysToCheck:
            s = gtsam.Symbol(k)
            old_chr = chr(s.chr())

            if old_chr in charMap.keys():
                if keepKeyIndex:
                    # Create new key with same index as old key
                    newKey = gtsam.Symbol(charMap[old_chr], s.index()).key()
                    if self.currentEstimate_.exists(newKey):
                        raise ValueError(f"RenameKeysInPlace: key {gtsam.Symbol(newKey).string()} already exists")
                else:
                    # Create new key with the new character and the next available index for that character
                    if charMap[old_chr] not in self.sessionKeys:
                        newKey = gtsam.Symbol(charMap[old_chr], 0).key()
                    else:
                        newKey = self.sessionKeys[charMap[old_chr]][-1] + 1
                

                # Update keyMap, timeMap, and newValues with the rekeyed value
                keyMap[k] = newKey
                timeMap[newKey] = timeMap[k]
                timeMap[k] = -self.params.fixedLagKeepFrames # Reset old key's timestamp to 0.0 so that it gets removed from the smoother
                newValues.insert(newKey, self.currentEstimate_.atPose3(k))

                # Remove old key from sessionKeys, pcds, and recent keys
                if old_chr in self.sessionKeys:
                    self.DecrementSize(k) # Remove old
                    self.IncrementSize(newKey) # Add new
                if k in self.pcds_:
                    self.pcds_[newKey] = self.pcds_.pop(k)
                    self.timestamps_[newKey] = self.timestamps_.pop(k)
                if k in recentKeysQueue:
                    recentKeysQueue[recentKeysQueue.index(k)] = newKey

        self.recentKeysQueue_.clear()
        self.recentKeysQueue_.extend(recentKeysQueue)
        assert list(self.recentKeysQueue_) == recentKeysQueue, "recent keys queue doesn't match what it is supposed to be"

        # Create new graph with rekeyed factors
        rekeyedFactors = gtsam.NonlinearFactorGraph()
        factorIndicesToRemove = []
        for i in range(graph.size()):
            f = graph.at(i)
            if not f:
                continue
            
            # Check if any of the factor's keys are in the charMap, and if so, rekey the factor and add it to the new graph
            for k in f.keys():
                if (True if k in keys else chr(gtsam.Symbol(k).chr()) in charMap.keys()): # TODO: Verify that it works
                    new_keys = [keyMap[k] if k in keyMap else k for k in f.keys()]
                    rekeyedFactors.add(f.rekey(new_keys))
                    factorIndicesToRemove.append(i)
                    break

        # Also need to unfreeze old nodes if frozen so that they will be marginalized out
        frozenKeys = self.smoother_.frozenKeys()
        keysToUnfreeze = list(set(keyMap.keys()) - set(frozenKeys))

        if freezeNewNodes:
            keysToFreeze = list(keyMap.values())
        
        # Update smoother with new graph and values, and remove old factors
        self.smoother_.update(rekeyedFactors, newValues, timeMap, factorIndicesToRemove, gtsam.KeySet(keysToFreeze), gtsam.KeySet(keysToUnfreeze))
        self.currentEstimate_ = self.smoother_.calculateEstimate()

        # If lastOdomKey_ was rekeyed, update it to the new key
        if self.lastOdomKey_ in keyMap:
            self.lastOdomKey_ = keyMap[self.lastOdomKey_]


    
    def LoadScans(self, directory:str|Path, symbolChr:str|None=None):
        pcds = fgtools.loadPosePCDs(str(directory))
        if symbolChr:
            for k, v in pcds.items():
                self.pcds_[gtsam.Symbol(symbolChr, k).key()] = v
        else:
            if self.priorValues_.size() > 0:
                for k, v in pcds.items():
                    self.pcds_[gtsam.Symbol(self.priorSessionSymbols_[0], k).key()] = v
            elif self.currentEstimate_.size() > 0:
                for k, v in pcds.items():
                    self.pcds_[gtsam.Symbol(self.currentSessionSymbol_, k).key()] = v
            else:
                self.pcds_ = pcds
        assert len(self.pcds_.keys()) == len(self.currentEstimate_.keys()), "Number of PCDs must match number of values in the graph. Check to make sure you loaded values first."


    def CreateGlobalPCD(self, color: tuple[float, float, float] | None = None, voxel_size: float | None = None, changeColorBySymbols: bool = False, symbolColors: dict[int, str] = {}) -> o3d.geometry.PointCloud:
        if not self.pcds_:
            raise Exception("Must load PCDs first before creating global point cloud")
        return fgtools.createGlobalPCD(self.pcds_, self.currentEstimate_, color=color, voxel_size=voxel_size, changeColorBySymbols=changeColorBySymbols, symbolColors=symbolColors)

    def AddFactor(self, factor: gtsam.gtsam.Factor, key:int|None = None, estimate: gtsam.gtsam.Pose3|None=None) -> None:
        self.graph_.add(factor)

        if key is not None and estimate is not None:
            self.values_.insert(key, estimate)

    def AddPrior(self, key: int, pose: gtsam.Pose3, noise: gtsam.noiseModel.Base, skipValue:bool=False):
        f = gtsam.PriorFactorPose3(key, pose, noise)
        if skipValue:
            self.AddFactor(f)
        else:
            self.AddFactor(f, key, pose)
            # Set initial odom key if needed
            if self.lastOdomKey_ == 0:
                self.lastOdomKey_ = key
            self.recentKeysQueue_.append(key)


        self.UpdateSmoother()


    
    def AddOdomFactor(self, pose: gtsam.Pose3|None=None, 
                            relPose:gtsam.Pose3|None=None, 
                            newPointCloud:o3d.geometry.PointCloud=o3d.geometry.PointCloud()
                            ) -> int:
        if self.currentEstimate_.empty():
            raise RuntimeError(f"Values is empty. Must add a prior value before adding odometry factor")
        
        if pose is None and relPose is None:
            raise ValueError("AddOdomFactor: Either pose or relPose must be provided")
        
        elif pose is None and relPose is not None:
            poseFrom = self.currentEstimate_.atPose3(self.lastOdomKey_)
            pose = poseFrom.compose(relPose)
        elif relPose is None and pose is not None:
            poseFrom = self.currentEstimate_.atPose3(self.lastOdomKey_)
            relPose = poseFrom.between(pose)
        
        decisionStateChanged = self.UpdateDecisionState()

        # Assumed first node was localization, it the first decision is map, change the first node
        if len(self.recentKeysQueue_) == 1 and self.decisionState_ == NodeDecision.MAP:
            assert self.currTimestamp_ == 1, "Changing prior. Timestamp should be 1.0"
            self.RenameKeysInPlace({self.currentSessionSymbol_: self.addToPriorSymbol_}, [self.recentKeysQueue_[0]])
            # decisionStateChanged = False
        # if self.initialFactorInfo_:
        #     symbol = self.addToPriorSymbol_ if self.decisionState_ == NodeDecision.MAP else self.currentSessionSymbol_
        #     key = gtsam.Symbol(symbol, 0).key()
        #     self.AddPrior(key, self.initialFactorInfo_[0], self.initialFactorInfo_[1])
        #     self.initialFactorInfo_.clear()

        # Can Use the following to refine the odometry match if needed
        if self.params.refineOdometry:
            submap = self.FindRecentKeyFrames(5)
            reg = fgtools.ScanMatch(newPointCloud, submap, pose.matrix(), showResult=0, maxCorrespondenceDist=2.0, voxelSize=self.params.icp.voxelSize)
            pose = gtsam.Pose3(reg.transformation)
            relPose = poseFrom.between(pose)

        # If not mapping, the key should be the fixed lag key
        if self.decisionState_ != NodeDecision.MAP:
            newKey = gtsam.Symbol(self.currentSessionSymbol_, int(self.currTimestamp_)).key()

        # If mapping, the key will be the new session key being added to the prior
        else:
            newKey = gtsam.Symbol(self.addToPriorSymbol_, int(self.currTimestamp_)).key()
        
        # noise = self.CalcOdomNoise(newPointCloud, pose)
        noise = self.odomNoise_
                
        factor = gtsam.BetweenFactorPose3(
            self.lastOdomKey_,
            newKey,
            relPose,
            noise
        )

        self.AddFactor(factor, newKey, pose)
        result = self.UpdateSmoother(optimizationTimeList=self.timeToOptimize_)

        deletedKeys = result.getKeysOfDeletedNodes()
        if deletedKeys:
            for k in deletedKeys:
                self.sessionKeys[chr(gtsam.Symbol(k).chr())].remove(k)
                self.pcds_.pop(k)
                self.timestamps_.pop(k)
                if k in self.currentToPriorAnchors_:
                    self.currentToPriorAnchors_.pop(k)

        if self.params.fixedLagMarg == FixedLagMarginalizationType.APPROX_PRIOR:
            self.ApproximateLinearFactor(result.getMarginalFactorIndices(), self.params.linearFactorMaxNodes)
        
        if decisionStateChanged:
            if self.decisionState_ == NodeDecision.MAP:
                self.mappingStartKeys_.append(newKey)
                if self.params.policyType == DecisionType.RELATIVE_COVARIANCE:
                    self.relCovStartKeys_.append(newKey)
            else:
                assert self.IsAddToPriorSessionKey(self.lastOdomKey_), f"Expected lastOdomKey_ to be an addToPriorSessionKey but is actually {gtsam.Symbol(self.lastOdomKey_).string()}"
                self.mappingEndKeys_.append(self.lastOdomKey_)
        self.recentKeysQueue_.append(newKey)
        self.lastOdomKey_ = newKey

        return newKey

    def CalcOdomNoise(self, newPointCloud:o3d.geometry.PointCloud, newPose:gtsam.Pose3) -> gtsam.noiseModel.Diagonal:
        if not newPointCloud.has_points():
            print("WARNING: New odom point cloud has no points. Returning default noise model.")
            return self.odomNoise_
        currentScan = o3d.geometry.PointCloud(newPointCloud)
        localSubmap = self.FindRecentKeyFrames(5)

        result = fgtools.ScanMatch(currentScan, localSubmap, newPose.matrix(),showResult=0, voxelSize=self.params.icp.voxelSize)
        # print(f"OdomNoise: fitness - {result.fitness}; inlier_rmse - {result.inlier_rmse}")
        if np.isclose(result.fitness, 1.0):
            raise ValueError("CalcOdomNoise: scan match fitness is 1.0. Very high likelihood that there's a bug.")
        # error = fgtools.GetMeanErrorFromScanMatchResult(currentScan, localSubmap, result)
        error = result.inlier_rmse
        iNoise = max((1 - result.fitness) * error, 1e-4)
        # print(f"Odom Noise: {iNoise}")
        return gtsam.noiseModel.Isotropic.Variance(6, iNoise)

    def AddGpsFactor(self, timestamp: float, lat: float, lon: float, alt: float, cov: np.ndarray, covType: int):

        # Wait for initialization
        if not self.recentKeysQueue_:
            return

        closestTimestampKey = None
        timestampDiff = np.inf
        for k in reversed(self.recentKeysQueue_):
            t = self.timestamps_[k]
            if abs(t - timestamp) < timestampDiff:
                timestampDiff = abs(t - timestamp)
                closestTimestampKey = k
            if t < timestamp:
                break

        if not closestTimestampKey or timestampDiff > 0.5:
            return

        closestPose = self.currentEstimate_.atPose3(closestTimestampKey)

        # Get reference for gps
        if not np.isfinite(self.refLatitude_) or not np.isfinite(self.refLongitude_):
            self.refLatitude_, self.refLongitude_ = fgtools.reference_latlon_from_xy(lat, lon, closestPose.x(), closestPose.y())
            self.refAltitude_ = alt - closestPose.z()
            ref_x, ref_y = fgtools.latlon_to_xy(lat, lon, self.refLatitude_, self.refLongitude_)
            ref_z = alt - self.refAltitude_
            print(f"Reference - given pose {closestPose.x(), closestPose.y(), closestPose.z()} gps {float(ref_x), float(ref_y), ref_z}")

        closestKeyCov = self.smoother_.marginalCovariance(closestTimestampKey)

        if closestKeyCov[3, 3] < self.params.gpsCovThreshold and closestKeyCov[4, 4] < self.params.gpsCovThreshold:
            return

        # Convert lat lon alt to x y z
        x, y = fgtools.latlon_to_xy(lat, lon, self.refLatitude_, self.refLongitude_)
        if self.params.dataset == "nclt": # For whatever reason the body frame is inverted
            xtmp = copy(x)
            x = copy(y)
            y = xtmp
        z = alt - self.refAltitude_

        gpsPoint = gtsam.Point3(x, y, z)

        if hasattr(self, "lastGpsPoint_"):
            if np.linalg.norm(self.lastGpsPoint_[:2] - gpsPoint[:2]) < self.params.gpsDistThreshold:
                return
        self.lastGpsPoint_ = gpsPoint 

        print(f"Pose x: {closestPose.x()} y: {closestPose.y()} - GPS x: {x} y: {y}")

        if covType  <= 1:
            gpsCov = np.eye(3)
        else:
            gpsCov = cov
            print(f"GPS cov known. Check to see if it's reasonable:\n{cov}")

        factor = gtsam.GPSFactor(closestTimestampKey, gpsPoint, gtsam.noiseModel.Gaussian.Covariance(gpsCov))
        factor.print()
        self.AddFactor(factor)
        print(f"Added gps factor on key {gtsam.Symbol(closestTimestampKey).string()}")
    
    def AddPointCloud(self, key: int, pointCloud: o3d.geometry.PointCloud|np.ndarray, timestamp: float) -> bool:
        if key in self.pcds_.keys():
            raise ValueError(f"Point cloud for key {gtsam.Symbol(key).string()} already exists.")
            return False
        
        if isinstance(pointCloud, np.ndarray):
            assert pointCloud.shape[1] == 3, f"Point cloud must be of shape (N, 3) but is actually {pointCloud.shape}"
            pointCloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointCloud))
        
        self.pcds_[key] = pointCloud
        self.timestamps_[key] = timestamp
        return True
    
    def SetPointClouds(self, scans: dict[int, o3d.geometry.PointCloud]) -> None:
        self.pcds_ = deepcopy(scans)

    def SetTimestamps(self, timestamps: dict[int, float]) -> None:
        self.timestamps_ = deepcopy(timestamps)
        if self.timestamps_.keys() != self.pcds_.keys():
            print(f"Warning: Timestamps keys do not match point cloud keys.")
    
    def UpdatePriorTree(self, calcRecentKeysToIgnore:bool=True):
        # 
        if self.addToPriorSymbol_ in self.sessionKeys:
            if self.params.priorUpdateMethod == 2:
                self.loopKeysToIgnore_ = set(self.recentKeysQueue_) & set(self.sessionKeys[self.addToPriorSymbol_])
            else:
                self.loopKeysToIgnore_ |= set(self.recentKeysQueue_) & set(self.sessionKeys[self.addToPriorSymbol_])
            # Remove the nodes that are in the current mapping section
            if self.mappingStartKeys_ and self.params.priorUpdateMethod == 1:
                keysInCurrentMapping = [k for k in self.loopKeysToIgnore_ if k >= self.mappingStartKeys_[-1]]
                self.loopKeysToIgnore_ -= set(keysInCurrentMapping)
        if calcRecentKeysToIgnore:
            potentialKeysToIgnore = set(self.sessionKeys[self.addToPriorSymbol_]) if self.addToPriorSymbol_ in self.sessionKeys else set()
            keysAlreadyInPrior = set(self.interLoopTree_.keys) if self.interLoopTree_ else set()
            self.loopKeysToIgnore_ |= potentialKeysToIgnore - keysAlreadyInPrior
        points, keys = fgtools.GetPointsFromGraphEstimate(self.currentEstimate_, 
                                                          symbolChr=self.currentSessionSymbol_, 
                                                          allOthers=True, 
                                                          keysToSkip=self.loopKeysToIgnore_
                                                          )

        if not points: # If points is empty
            print("Warning: Can't updatePriorTree because data points is empty.")
            return
        
        # if len(self.loopKeysToIgnore_):
        #     tmpPoints, tmpKeys = [], []
        #     for p, k in zip(points, keys):
        #         if k not in self.loopKeysToIgnore_:
        #             tmpPoints.append(p)
        #             tmpKeys.append(k)
        #     points, keys = tmpPoints, tmpKeys

        tree = KDTree(points)
        self.interLoopTree_ = TreeData(tree, keys, points)
    
    # def UpdatePriorKeyToFactorIndexDict(self):
    #     self.priorKeyToFactorIndexDict_.clear()

    #     for i in range(self.priorGraph_.size()):
    #         f = self.priorGraph_.at(i)
    #         keys = f.keys()
    #         for k in keys:
    #             fgtools.AddValueToDictList(self.priorKeyToFactorIndexDict_, k, i)
        
    
    def CheckLoopClosures(self, intra:bool=True, inter:bool=True) -> tuple[int, int]:
        '''
        Check for loop closures if specified (default=True) and optimize the graph
        '''
        currPose = self.currentEstimate_.atPose3(self.lastOdomKey_)

        # Update prior session as needed
        if self.params.priorUpdateMethod > 0:
            if self.decisionState_ == NodeDecision.MAP:
                if self.params.priorUpdateMethod == 2:
                    self.UpdatePriorTree(calcRecentKeysToIgnore=False)
                elif len(self.loopKeysToIgnore_) and self.params.priorUpdateMethod == 1:
                    self.UpdatePriorTree(calcRecentKeysToIgnore=False)
            elif len(self.loopKeysToIgnore_) and self.params.priorUpdateMethod == 1:
                self.UpdatePriorTree(calcRecentKeysToIgnore=False)


        # Intra loop closures (same session)
        numIntraLoops = 0
        if intra and self.decisionState_ == NodeDecision.MAP:
            points, keys = fgtools.GetPointsFromGraphEstimate(self.currentEstimate_, 
                                                                [self.currentSessionSymbol_, self.addToPriorSymbol_], 
                                                                keysToSkip=set(self.recentKeysQueue_))
            if points:
                points = np.asarray(points)
                tree = KDTree(points)
                distances, indices = tree.query(currPose.translation(), k=self.params.loopClosure.intraNearKeyFrames, distance_upper_bound=self.params.loopClosure.distMax) # Look for the 5 closest poses given the distance is less than 30
                # distances, indices = tree.query(currPose.translation())  # Look for closest pose

                keysToSearch = np.asarray(keys)[indices[distances < self.params.loopClosure.distMax]]
                # for k in keysToSearch:
                foundIntraLoop, _ = self.LoopCheckGivenKey(keysToSearch)
                if foundIntraLoop:
                    self.intraLoopKeys_.append(self.lastOdomKey_)
                    numIntraLoops += 1

        # Inter loop closures
        numInterLoops = 0
        if inter and self.interLoopTree_:

            distances, indices = self.interLoopTree_.tree.query(currPose.translation(), 
                                                                k=self.params.loopClosure.interNearKeyFramesPrior) 
            # distances, indices = self.interLoopTree_.tree.query(pose.translation()) # Look for the 5 closest poses given the distance is less than 30
            keys = deepcopy(self.interLoopTree_.keys)

            keysToSearch = np.asarray(keys)[indices[distances < self.params.loopClosure.distMax]]

            numInterLoops = 0

            pcSource = self.FindRecentKeyFrames(0)
            self.closestSubmapData_.currScan = o3d.geometry.PointCloud(pcSource)
            self.closestSubmapData_.currCenterPoint = self.currentEstimate_.atPose3(self.lastOdomKey_).translation()
            self.closestSubmapData_.currKey = copy(self.lastOdomKey_)

            # for i, k in enumerate(keysToSearch):
                # ret = self.LoopCheckGivenKey(k, addFactor=True, numNearKeyFrames=self.params.icp.nearKeyFrames)
            foundInterLoop, reg = self.LoopCheckGivenKey(keysToSearch, 
                                                        addFactor=True, 
                                                        numNearSourceKeyFrames=self.params.loopClosure.interNearKeyFramesCurrent, 
                                                        setSubmap=True, 
                                                        isInter=True
                                                        )
            if foundInterLoop:
                numInterLoops += 1
                
            self.lastInterICPResult_ = reg
        if numIntraLoops > 0 or numInterLoops > 0:
            self.UpdateSmoother()
        return numIntraLoops, numInterLoops

    def LoopCheckGivenKey(self, keys: list[int], 
                          addFactor:bool=True, 
                          numNearSourceKeyFrames: int = 0, 
                          setSubmap: bool = False,
                          isInter: bool = False
                          ) -> tuple[bool, o3d.pipelines.registration.RegistrationResult]:

        foundLoop = False
        if len(keys) == 0:
            return foundLoop, None
        
        currKey = self.lastOdomKey_

        pcSource = self.FindRecentKeyFrames(numNearSourceKeyFrames)
        # pcSource = self.FindRecentKeyFrames(1)
        # pcTarget = self.FindSequentialKeyFrames(keys[0], numNearTargetKeyFrames)
        pcTarget = self.GetSubmapFromKeys(keys)

        if self.currentEstimate_.exists(keys[0]):
            poseTarget = self.currentEstimate_.atPose3(keys[0])
        elif self.priorValues_.exists(keys[0]):
            poseTarget = self.priorValues_.atPose3(keys[0])
        else:
            raise Exception(f"Key {gtsam.Symbol(keys[0]).string()} is not found in currentEstimate of priorValues")

        if setSubmap:
            self.closestSubmapData_.closestSubmap = pcTarget
            self.closestSubmapData_.submapCenterPoint = poseTarget.translation()
            self.closestSubmapData_.submapKey = copy(keys[0])

        showResult = False
        if self.currTimestamp_ % (self.params.loopClosure.showResultSkipFrames + 1) == 0:
            showResult = self.params.loopClosure.showScanMatchResult

        reg = fgtools.ScanMatch(pcSource, pcTarget, np.eye(4), showResult=showResult, 
                                maxCorrespondenceDist=0.5 if isInter else self.params.loopClosure.maxCorrespondenceDist,
                                voxelSize=self.params.loopClosure.voxelSize, 
                                returnTightEval=True
                                )

        fit = reg.fitness
        rmse = reg.inlier_rmse
        transFinal = reg.transformation
        if self.params.loopClosure.debug:
            print(f"Loop Check between keys {gtsam.Symbol(currKey).string()} and {gtsam.Symbol(key).string()} fitness: {fit:.4f} inliers: {rmse:.4f} correspondence len: {len(reg.correspondence_set)}")

        if fit > self.params.loopClosure.fitnessThreshold and rmse < self.params.loopClosure.rmseThreshold:
            foundLoop = True
            cs = gtsam.Symbol(currKey)
            ts = gtsam.Symbol(keys[0])
            if self.params.loopClosure.debug:
                print(f"Loop Found between keys {chr(cs.chr()) if cs.chr() else ''}{cs.index()} and {chr(ts.chr()) if ts.chr() else ''}{ts.index()}")
            # loopNoise = gtsam.noiseModel.Diagonal.Variances(np.asarray([max(1-fit, 1e-2) * self.params.loopClosure.noiseScale] * 6))

            # error = fgtools.GetMeanErrorFromScanMatchResult(pcSource, pcTarget, reg)
            error = reg.inlier_rmse
            noiseValue = max((1-fit)*error, 1e-3)
            loopNoise = gtsam.noiseModel.Isotropic.Variance(6, noiseValue)
            # print(f"Loop Noise: {noiseValue}")

            Twrong = self.currentEstimate_.atPose3(currKey).matrix()

            Tfixed = transFinal @ Twrong
            posefixed = gtsam.Pose3(Tfixed)
            if addFactor:
                self.AddFactor(gtsam.BetweenFactorPose3(currKey, keys[0], posefixed.between(poseTarget), loopNoise))
                if self.IsCurrentSessionKey(currKey) and not self.IsCurrentSessionKey(keys[0]):
                    fgtools.AddValueToDictList(self.currentToPriorAnchors_, currKey, keys[0])

        return foundLoop, reg
    
    def GetSubmapFromKeys(self, keys:list[int], downsample:bool=False) -> o3d.geometry.PointCloud:
            localMap = o3d.geometry.PointCloud()
    
            for k in keys:
                if self.currentEstimate_.exists(k):
                    transformation = self.currentEstimate_.atPose3(k).matrix()
                elif self.priorValues_.exists(k):
                    transformation = self.priorValues_.atPose3(k).matrix()
                else:
                    continue
                pcd = deepcopy(self.pcds_[k])
                pcd.transform(transformation)
                localMap += pcd
            
            if downsample:
                localMap = localMap.voxel_down_sample(0.5) # voxelize localMap
    
            return localMap
    
    def FindSequentialKeyFrames(self, key:int, searchNum:int, downsample:bool=False) -> o3d.geometry.PointCloud:
        localMap = o3d.geometry.PointCloud()

        searchKeys = np.arange(-searchNum, searchNum+1)
        for i in searchKeys:
            keyNear = int(key + i)
            if self.currentEstimate_.exists(keyNear):
                transformation = self.currentEstimate_.atPose3(keyNear).matrix()
            elif self.priorValues_.exists(keyNear):
                transformation = self.priorValues_.atPose3(keyNear).matrix()
            else:
                continue
            pcd = deepcopy(self.pcds_[keyNear])
            pcd.transform(transformation)
            localMap += pcd
        
        if downsample:
            localMap = localMap.voxel_down_sample(0.5) # voxelize localMap

        return localMap

    def FindRecentKeyFrames(self, searchNum:int) -> o3d.geometry.PointCloud:
        if searchNum < 0:
            raise ValueError(f"FindRecentKeyFrames: searchNum must be greater than or equal to 0, but is {searchNum}")
        localMap = o3d.geometry.PointCloud()

        if (searchNum+1) < len(self.recentKeysQueue_):
            searchKeys = list(self.recentKeysQueue_)[-(searchNum+1):]
        else:
            searchKeys = list(self.recentKeysQueue_)

        for key in searchKeys:
            if self.currentEstimate_.exists(key):
                transformation = self.currentEstimate_.atPose3(key).matrix()
            elif self.priorValues_.exists(key):
                transformation = self.priorValues_.atPose3(key).matrix()
            else:
                continue
            pcd = deepcopy(self.pcds_[key])
            pcd.transform(transformation)
            localMap += pcd

        return localMap

    def Size(self, symbolChr:str|None=None) -> int:
        '''
        If a specific symbol is provided, return the number of nodes using that symbol.
        If input is None (default), the size of the entire graph is returned.
        '''
        if symbolChr is not None:
            if symbolChr in self.sessionKeys:
                return len(self.sessionKeys[symbolChr])
            else:
                return 0
        else:
            totalSize = 0
            for v in self.sessionKeys.values():
                totalSize += len(v)
            return totalSize
        

    def IncrementSize(self, keys: int|list[int]) -> None:
        if isinstance(keys, int):
            keys = [keys]

        for key in keys:
            s = gtsam.Symbol(key)
            if s.chr():
                char = chr(s.chr())
            else:
                char = ''
            if char in self.sessionKeys:
                self.sessionKeys[char].append(key)
            else:
                self.sessionKeys[char] = [key]

    def DecrementSize(self, keys: int|list[int]) -> None:
        if isinstance(keys, int):
            keys = [keys]
        
        for key in keys:
            s = gtsam.Symbol(key)
            if s.chr():
                char = chr(s.chr())
            else:
                char = ''
            if char in self.sessionKeys:
                self.sessionKeys[char].remove(key)
                if len(self.sessionKeys[char]) == 0:
                    self.sessionKeys.pop(char)
            else:
                raise ValueError(f"DecrementSize: Key {gtsam.Symbol(key).string()} not found in sessionKeys")
    
    def ContainsKeyWithSymbols(self, keys: list[int], symbol:str|list[str]) -> bool:
        if isinstance(symbol, str):
            symbol = [symbol]
        symbolInts = [ord(s) for s in symbol]
        for k in keys:
            if gtsam.Symbol(k).chr() in symbolInts:
                return True
        return False
    
    def IsCurrentSessionKey(self, key:int) -> bool:
        s = gtsam.Symbol(key)
        if s.chr() == ord(self.currentSessionSymbol_):
            return True
        return False
    
    def ContainsCurrentSessionKey(self, keys: list[int]) -> bool:
        for k in keys:
            if self.IsCurrentSessionKey(k):
                return True
        return False

    def IsPriorSessionKey(self, key:int) -> bool:
        s = gtsam.Symbol(key)
        if chr(s.chr()) in self.priorSessionSymbols_:
            return True
        return False
    
    def IsAddToPriorSessionKey(self, key:int) -> bool:
        s = gtsam.Symbol(key)
        if s.chr() == ord(self.addToPriorSymbol_):
            return True
        return False

    def SetGpsReference(self, lat, lon, alt):
        self.refLatitude_ = lat
        self.refLongitude_ = lon
        self.refAltitude_ = alt
    
    def GetSessionInfo(self, excludeSymbols: list[str]|None = None) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values, dict[int, o3d.geometry.PointCloud], dict[int, float]]:
        if excludeSymbols is None:
            excludeSymbols = []

        # Remove all current session keys from the graph, values, pcds, and timestamps
        infoGraph = gtsam.NonlinearFactorGraph()
        tmpGraph = self.smoother_.getFactors()
        for i in range(tmpGraph.size()):
            f = tmpGraph.at(i)
            if not f:
                continue
            if not self.ContainsKeyWithSymbols(f.keys(), excludeSymbols):
                infoGraph.push_back(f)

        infoValues = gtsam.Values()
        infoPcds = {}
        infoTimestamps = {}
        for k in self.values.keys():
            s = gtsam.Symbol(k).chr()
            if chr(s) not in excludeSymbols:
                infoValues.insert(k, self.values.atPose3(k))
                infoPcds[k] = self.pcds_[k]
                infoTimestamps[k] = self.timestamps_[k]

        return infoGraph, infoValues, infoPcds, infoTimestamps
    
    def GetFinalSessionInfo(self) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values, dict[int, o3d.geometry.PointCloud], dict[int, float]]:
        '''
        DENSE: Marginalize out all current nodes before calling GetSessionInfo.
        APPROX: Remove all current nodes and factors before calling GetSessionInfo.
        '''
        if self.params.fixedLagMarg == FixedLagMarginalizationType.DENSE:
            timestamps = {}
            for k in self.sessionKeys[self.currentSessionSymbol_]:
                timestamps[k] = -self.params.fixedLagKeepFrames - 1

            result = self.smoother_.update(gtsam.NonlinearFactorGraph(), gtsam.Values(), timestamps)
            
            deletedKeys = result.getKeysOfDeletedNodes()
            if deletedKeys:
                for k in deletedKeys:
                    self.sessionKeys[chr(gtsam.Symbol(k).chr())].remove(k)
                    self.pcds_.pop(k)
                    self.timestamps_.pop(k)
                    if k in self.currentToPriorAnchors_:
                        self.currentToPriorAnchors_.pop(k)

            return self.GetSessionInfo()
        else:
            return self.GetSessionInfo(excludeSymbols=[self.currentSessionSymbol_])
 
        

    def SetPriorSessionInfo(self, graph:gtsam.NonlinearFactorGraph, values:gtsam.Values, pcds:dict[int, o3d.geometry.PointCloud], timestamps:dict[int, float]):
        '''
        Set the prior session information for the SessionManager. Resets the session to the prior session and clears the current session.
        param graph: The prior session graph
        param values: The prior session values
        param pcds: The prior session point clouds
        param timestamps: The prior session timestamps
        '''

        self.priorGraph_ = graph
        self.priorValues_ = values
        self.pcds_ = pcds
        self.timestamps_ = timestamps

        if self.params.rekeyPrior:
            self.priorGraph_, self.priorValues_ = self.RenameKeys(chars=self.params.priorSessionSymbolChrs,
                                                                  autoRename=self.params.autoRename,
                                                                  updateOptimization=False,
                                                                  )
        self.UpdateSmoother(reset=True)
        self.UpdatePriorTree()
        # self.UpdatePriorKeyToFactorIndexDict()

    def SaveSession(self, directory:str|Path|None=None):
        if directory is None:
            directory = self.params.saveSessionDir
        
        if self.params.loadSessionDir:
            directory = Path(directory) / self.params.policyType.name
        else:
            directory = Path(directory)

        while directory.exists():
            res = input(f"Directory {directory} already exists. Overwrite? (y/[n]): ")
            if res and res.lower()[0] == "y":
                print("Removing existing directory and overwriting...")
                shutil.rmtree(directory)
            else:
                tmp_directory = copy(directory)
                while tmp_directory == directory:
                    res = input(f"Enter a new directory path or enter n/no to abort: ")
                    if res.lower() == "n" or res.lower() == "no":
                        print("Aborting save...")
                        return
                    else:
                        confirm = input(f"Confirm new directory path {res} (y/[n]): ")
                        if confirm and confirm.lower()[0] == "y":
                            directory = Path(res)
        print(f"Saving session to {directory}...")
        os.makedirs(directory, exist_ok=True)

        finalGraph, finalValues, finalPcds, finalTimestamps = self.GetFinalSessionInfo()
        # fgtools.writeG2oWithPrior(self.graph, self.values, str(directory / "graph.g2o"))

        # PCDs are stored once in the base directory (parent of the policy dir)
        # under scans/{A,B,C}/ so they can be shared across policies.
        if self.params.loadSessionDir:
            scans_root = directory.parent / "scans"
        else:
            scans_root = directory / "scans"
        
        if not fgtools.saveSymbolPCDs(finalPcds, scans_root):
            print(f"Error saving point clouds to {str(scans_root)}")
        if not gtsam_unstable.serializeGraphToFile(finalGraph, str(directory / "graph.gtsam")):
            print(f"Error saving graph to {str(directory / 'graph.gtsam')}")
        if not gtsam_unstable.serializeValuesToFile(finalValues, str(directory / "values.gtsam")):
            print(f"Error saving values to {str(directory / 'values.gtsam')}")
        if not fgtools.saveOdomTUM(finalTimestamps, finalValues, str(directory / "odom_tum.txt")):
            print(f"Error saving odometry to {str(directory / 'odom_tum.txt')}")
        if not fgtools.saveTimestampsWithKeys(finalTimestamps, str(directory / "timestamps.txt")):
            print(f"Error saving timestamps to {str(directory / 'timestamps.txt')}")
        if not fgtools.SaveListToFile(self.timeToCalcMetricsAndDecision_, directory / "timeToCalcMetricsAndDecision.txt"):
            print(f"Error saving timeToCalcMetricsAndDecision to {str(directory / 'timeToCalcMetricsAndDecision.txt')}")
        if not fgtools.SaveListToFile(self.timeToOptimize_, directory / 'timeToOptimize.txt'):
            print(f"Error saving timeToOptimize to {str(directory / 'timeToOptimize.txt')}")

        # Save params to metadata file
        metadata = fgtools.make_yaml_safe(self.params)
        with open(directory / "metadata.yaml", "w") as f:
            yaml.safe_dump(metadata, f, sort_keys=False)

        print(f"Done saving session.")

    def LoadSession(self, directory:str|Path|None=None, sharedScans: dict[int, o3d.geometry.PointCloud] | None = None):

        if directory is None:
            directory = self.params.loadSessionDir
            if directory is None:
                raise ValueError("LoadSession: No directory provided and params.loadSessionDir is None.")

        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        
        graphFile = directory / "graph.gtsam"
        valuesFile = directory / "values.gtsam"
        odomFile = directory / "odom_tum.txt"
        timestampsFile = directory / "timestamps.txt"

        # Detect whether this is the new shared-scans layout:
        # scans/ lives in the parent (base) dir and has single-letter subdirs.
        # baseScansDir = directory.parent / "scans"
        # legacyScansDir = directory / "scans"
        # sharedScanSymbols = fgtools.discoverScanSymbols(baseScansDir)
        # useSharedScans = bool(sharedScanSymbols)

        # scansDir = baseScansDir if useSharedScans else legacyScansDir

        if Path(directory.parent / "scans").exists():
            scansDir = directory.parent / "scans"
        elif Path(directory / "scans").exists():
            scansDir = directory / "scans"
        else:
            raise ValueError(f"LoadSession: Can't find 'scans' directory in directory or its parent - {directory}")

        if not graphFile.exists() or not valuesFile.exists() or not scansDir.exists() or not odomFile.exists() or not timestampsFile.exists():
            raise FileNotFoundError(f"One or more files in {directory} are missing. Required files: graph.gtsam, values.gtsam, scans/, odom_tum.txt, timestamps.txt")

        self.priorGraph_ = gtsam_unstable.deserializeGraphFromFile(str(graphFile))
        self.priorValues_ = gtsam_unstable.deserializeValuesFromFile(str(valuesFile))
        self.timestamps_ = fgtools.loadTimestampsWithKeys(str(timestampsFile))

        # if useSharedScans:
            # New layout: keys on disk already use the correct symbol characters.
            # Set prior/addToPrior from the folders that are actually represented
            # in the loaded values; no RenameKeys needed.
            # if sharedScanSymbols:
            #     self.addToPriorSymbol_ = chr(ord(sharedScanSymbols[-1]) + 1)
            # else:
            #     raise RuntimeError("LoadSession: sharedScanSymbols is empty.")
            #     self.addToPriorSymbol_ = 'A'

        sharedScanSymbols = fgtools.discoverScanSymbols(scansDir)
        
        self.pcds_ = fgtools.loadSymbolPCDs(scansDir, keys=list(self.priorValues_.keys()), sharedScans=sharedScans)
        if not self.pcds_:
            raise ValueError("LoadSession: pcds after loading is still empty!")
        
        self.priorSessionSymbols_ = deepcopy(sharedScanSymbols)
        if not self.priorSessionSymbols_:
            self.priorSessionSymbols_ = set([chr(gtsam.Symbol(k).chr()) for k in self.pcds_.keys()])

            # Legacy flat layout: full GTSAM key as filename, may need RenameKeys.
            # self.pcds_ = fgtools.loadPosePCDs(str(scansDir))
        if self.priorSessionSymbols_:
            self.addToPriorSymbol_ = chr(ord(list(self.priorSessionSymbols_)[-1]) + 1)
        else:
            raise RuntimeError("LoadSession: sharedScanSymbols is empty.")
            self.addToPriorSymbol_ = 'A'

        valuesKeys = set(self.priorValues_.keys())
        if valuesKeys != set(self.pcds_.keys()) or valuesKeys != set(self.timestamps_.keys()):
            raise ValueError(f"LoadSession: Keys in values, scans, and timestamps do not match. \
                                # Value keys: {len(self.priorValues_.keys())}, # Scan keys: {len(self.pcds_.keys())}, \
                                # Timestamp keys: {len(self.timestamps_.keys())}")

        if not sharedScanSymbols:

            if self.params.rekeyPrior:
                self.priorGraph_, self.priorValues_ = self.RenameKeys(chars=self.params.priorSessionSymbolChrs,
                                                                      autoRename=self.params.autoRename,
                                                                      updateOptimization=False)

        self.UpdateSmoother(reset=True)
        self.UpdatePriorTree()
        # self.UpdatePriorKeyToFactorIndexDict()

        valuesKeys = set(self.priorValues_.keys())
        if valuesKeys != set(self.pcds_.keys()) or valuesKeys != set(self.timestamps_.keys()):
            raise ValueError(f"LoadSession: Keys in values, scans, and timestamps do not match after loading. " \
                             f"# Value keys: {len(self.priorValues_.keys())}, # Scan keys: {len(self.pcds_.keys())}, " \
                             f"# Timestamp keys: {len(self.timestamps_.keys())}")


    def Update(self, pointCloud: o3d.geometry.PointCloud, 
                     pose:np.ndarray|gtsam.gtsam.Pose3|None=None, 
                     relPose:gtsam.Pose3|None=None, 
                     timestamp:float=0.0,
                     gpsData:list=[]
                     ) -> UpdateResult:
        '''
        Assumes pose is being added to last value in graph
        '''
        print(f"\nTime: {self.currTimestamp_}")

        result = self.UpdateResult()
        if isinstance(pose, np.ndarray):
            pose = pose.flatten()
            assert len(pose) == 7, "Pose must be in format [tx, ty, tz, qw, qx, qy, qz]"
            pose = gtsam.Pose3(gtsam.Rot3(pose[3], pose[4], pose[5], pose[6]), pose[:3])

        if self.decisionState_ == NodeDecision.INITIALIZE:
            if pose is None:
                raise ValueError("Pose must be provided for the first keyframe")
            key = self.Initialize(pointCloud, pose, timestamp)
        else:
            key = self.AddOdomFactor(pose, relPose, pointCloud)
            self.AddPointCloud(key, pointCloud, timestamp=timestamp)

        if self.params.useGps and gpsData and self.decisionState_ == NodeDecision.MAP:
            self.AddGpsFactor(*gpsData)

        result.key = key
        self.loopCnt_ += 1
        if self.loopCnt_ > self.params.loopClosure.skipFrames:
            self.loopCnt_ = 0
            result.numIntraLoops, result.numInterLoops = self.CheckLoopClosures()

        # May need to rekey tail of window if in relative covariance mode
        if self.params.policyType == DecisionType.RELATIVE_COVARIANCE and self.relCovStartKeys_:
            if self.sessionKeys[self.currentSessionSymbol_]:
                relCovStartKeyIdex = gtsam.Symbol(self.relCovStartKeys_[0]).index()
                lastCurrKey = self.sessionKeys[self.currentSessionSymbol_][0]
                if gtsam.Symbol(lastCurrKey).index() < relCovStartKeyIdex:
                    if not self.interLoopTree_:
                        raise ValueError("Update: interloopTree is None")
                    _, index = self.interLoopTree_.tree.query(self.currentEstimate_.atPose3(lastCurrKey).translation())
                    closestPriorKey = self.interLoopTree_.keys[index]
                    jointCov = self.smoother_.getISAM2().jointMarginalCovariance([lastCurrKey, closestPriorKey]).fullMatrix()
                    logDet = np.log(np.linalg.det(fgtools.getRelativeCov(self.currentEstimate_.atPose3(lastCurrKey), self.currentEstimate_.atPose3(closestPriorKey), jointCov)))
                    if logDet > self.params.relativeCovariancePolicy.logDetThreshold:
                        # Rekey tail of window
                        keysToRekey = []
                        for k in self.sessionKeys[self.currentSessionSymbol_]:
                            if gtsam.Symbol(k).index() >= relCovStartKeyIdex:
                                break
                            keysToRekey.append(k)
                        self.RenameKeysInPlace({self.currentSessionSymbol_: self.addToPriorSymbol_}, keysToRekey)
                        self.relCovStartKeys_.pop(0)
                else:
                    self.relCovStartKeys_.pop(0)

            else:
                self.relCovStartKeys_.pop(0)


        # self.CheckDeterminants()
        self.currTimestamp_ += 1

        return result

    def Initialize(self, pointCloud: o3d.geometry.PointCloud, pose: gtsam.Pose3, timestamp: float):
        print("Initializing...")

        self.decisionState_ = NodeDecision.MAP

        # Scan match against prior map if one exists
        # if self.priorGraph_.size() != 0 and self.interLoopTree_:
        #     dist, idx = self.interLoopTree_.tree.query(pose.translation())
        #     nearestPriorKey = self.interLoopTree_.keys[idx]
        #     submap = self.FindSequentialKeyFrames(nearestPriorKey, 10)
        #     reg = fgtools.ScanMatch(pointCloud, submap, pose.matrix(), maxCorrespondenceDist=5.0, showResult=0, returnTightEval=True)
        #     if reg.fitness > self.params.loopClosure.fitnessThreshold:
        #         pose = gtsam.Pose3(reg.transformation)
        #         self.decisionState_ = NodeDecision.LOCALIZE
        #     else:
        #         print("\n****************************************************************************************")
        #         print("   Warning! Could not initialize with scan matching. Please check if this is correct!")
        #         print("****************************************************************************************\n")

        if self.priorGraph_.size() != 0:
            noise = gtsam.noiseModel.Diagonal.Variances(np.array([np.pi**2, np.pi**2, np.pi**2, 10, 10, 10]))
        else:
            noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2]*6))
        key = gtsam.Symbol(self.currentSessionSymbol_, 0).key()
        self.AddPrior(key, pose, noise)
        self.AddPointCloud(key, pointCloud, timestamp=timestamp)

        return key

    def CheckDeterminants(self):

        # Single Session traverse same area or multi-session rejoin prior map
        if self.intraLoopKeys_:
            start = time.time()
            firstKey = self.intraLoopKeys_[0]
            # If the new nodes are still forming loop closures, see if they are redundant
            if self.intraLoopKeys_[-1] == self.lastOdomKey_:
                self.intraLoopReset_ = 0
                self.intraLoopDets_.append(np.linalg.det(self.smoother_.marginalCovariance(firstKey)))

                if len(self.intraLoopDets_) >= 2:
                    pctChange = (self.intraLoopDets_[-2] - self.intraLoopDets_[-1]) / self.intraLoopDets_[-2]
                    print(f"Percent change determinant for key {gtsam.Symbol(firstKey).string()} is {pctChange*100:.2f}%")

                    if abs(pctChange) < 0.05:
                        self.ResetMode(NodeDecision.MAP)
                        print("Setting Mode to Localization")
                        self.decisionState_ = NodeDecision.LOCALIZE
                        # self.UpdatePriorTree()
            # Reset the intra loop keys if the new nodes aren't being loop closed.
            else:
                self.intraLoopReset_ += 1
                if self.intraLoopReset_ > 5:
                    self.ResetMode(NodeDecision.MAP)
            print(f"Time to calculate dets {time.time() - start}")

        slidingWindowKeys = self.GetMostRecentKeys(self.params.fixedLagKeepFrames + 1)
        # Leave localization and enter mapping
        for k in self.mappingStartKeys_:
            if k not in slidingWindowKeys:
                self.mappingStartKeys_.remove(k)
                self.mappingStartKeyDets_.pop(k)
                continue
            
            fgtools.AddValueToDictList(self.mappingStartKeyDets_, k, np.linalg.det(self.smoother_.marginalCovariance(k)))

            if len(self.mappingStartKeyDets_[k]) >= 2:
                pctChange = (self.mappingStartKeyDets_[k][-2] - self.mappingStartKeyDets_[k][-1]) / self.mappingStartKeyDets_[k][-2]
                print(f"Percent change determinant for key {gtsam.Symbol(k).string()} is {pctChange*100:.2f}%")

                if abs(pctChange) > 0.05:
                    # Relabel and freeze current session factors and values previous to k
                    keysToChange = []
                    for swk in slidingWindowKeys:
                        if swk == k:
                            break
                        if self.IsCurrentSessionKey(swk):
                            keysToChange.append(swk)
                    self.RenameKeysInPlace({self.currentSessionSymbol_: self.addToPriorSymbol_}, keepKeyIndex=True, freezeNewNodes=True)
                    self.mappingStartKeys_.remove(k)
                    self.mappingStartKeyDets_.pop(k)
                    

    def GetMostRecentKeys(self, numKeys:int) -> list[int]:
        mostRecentKeys = list(self.recentKeysQueue_)
        if len(mostRecentKeys) > numKeys:
            mostRecentKeys = mostRecentKeys[-numKeys:]
        return mostRecentKeys

    def GetKeys(self, symbolChr:str|None=None) -> list[int]:
        if symbolChr is None:
            return list(self.currentEstimate_.keys())
        else:
            if symbolChr in self.sessionKeys:
                return self.sessionKeys[symbolChr]
            else:
                return []

    @property
    def priorGraph(self):
        return self.priorGraph_
    
    @property
    def priorValues(self):
        return self.priorValues_
    
    @property
    def priorISAM2(self):
        priorISAM = gtsam.ISAM2()
        priorISAM.update(self.priorGraph_, self.priorValues_)
        priorISAM.update()
        return priorISAM
    
    @property
    def graph(self):
        return fgtools.cleanFactorGraph(self.smoother_)
        
    @property
    def values(self):
        return self.smoother_.calculateEstimate()
    
    @property
    def fixedLagSmoother(self):
        return self.smoother_

    @property
    def isam2(self):
        return self.smoother_.getISAM2()

    @property
    def scans(self) -> dict[int, o3d.geometry.PointCloud]:
        return self.pcds_
    
    @property
    def timestamps(self) -> dict[int, float]:
        return self.timestamps_
    
    @property
    def currentSessionSymbol(self) -> str:
        return deepcopy(self.currentSessionSymbol_)
    
    @property
    def addToPriorSymbol(self) -> str:
        return deepcopy(self.addToPriorSymbol_)

    @property
    def priorSessionSymbols(self) -> list[str]:
        return deepcopy(list(self.priorSessionSymbols_))
    
    @property
    def lastPose(self) -> gtsam.gtsam.Pose3:
        if self.currentEstimate_.exists(self.lastOdomKey_):
            return self.currentEstimate_.atPose3(self.lastOdomKey_)
        else:
            raise ValueError(f"Last odom key {gtsam.Symbol(self.lastOdomKey_).string()} not found in currentEstimate or priorValues")