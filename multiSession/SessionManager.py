#!/usr/bin/env python3

import numpy as np
np.set_printoptions(linewidth=np.inf, suppress=True)
from copy import copy, deepcopy
from pathlib import Path
import open3d as o3d
from scipy.spatial import KDTree

# Local imports
import common.FGVizHelpers as fgtools
import gtsam
from params import Params
from common.customStructs import TreeData
from metrics import Metrics
from policy.base import DecisionType, NodeDecision, NodeDecisionPolicy
from policy.naive import DistancePolicy, FitnessPolicy

class SessionManager:
    def __init__(self, filename:str|Path|None=None, loadGPS:bool=True, params:Params=Params()):
        self.params = params
        self.orig_graph_ = gtsam.NonlinearFactorGraph()
        self.orig_values_ = gtsam.Values()
        self.graph_ = gtsam.NonlinearFactorGraph()
        self.values_ = gtsam.Values()

        # I have gotten weird errors where calculateEstimate() tries to find a value I've already removed and throws an error.
        # I'm not sure why, but it works with these parameters.
        self.isamParams_ = gtsam.ISAM2Params()
        self.isamParams_.setRelinearizeThreshold(0.0) # Must be zero to work
        self.isamParams_.relinearizeSkip = 2 # Must be small (<4) to work
        self.isamParams_.findUnusedFactorSlots = True # Docs say to use this when doing fixedLagSmoother type things.
        # self.isamParams_.evaluateNonlinearError = True

        self.isam_ = gtsam.ISAM2(self.isamParams_)
        self.currentEstimate_ = gtsam.Values()

        self.pcds_: dict[int, o3d.geometry.PointCloud] = {}
        self.tree_: dict[str, KDTree] = {}


        self.betweenSessionLoopClosureIndices_: dict[int, list[int]] = {}

        self.decisionState = NodeDecision.LOCALIZE
        if self.params.decisionMethod == DecisionType.DISTANCE:
            self.decider = DistancePolicy(self.params)
        elif self.params.decisionMethod == DecisionType.FITNESS:
            self.decider = FitnessPolicy(self.params)
        else:
            print("SessionManager: Invalid decision policy chosen.")
            exit()
        self.metrics = Metrics(self.params.decisionMethod)

        self.priorTree_ = None
        self.addNodesToPrior_ = True
        self.addMultiSessionToPrior_ = False

        self.loopKeysToIgnore: set[int]  = set()
        self.lastInterICPResult_ = None
        
        self.currSessionSymbol_ = self.params.currSessionSymbolChr
        self.addToPriorSymbol_ = self.params.addToPriorSymbolChr
        self.priorSessionSymbols_ = self.params.priorSessionSymbolChrs

        self.sessionKeys: dict[str, list[int]] = {}

        self.isInitialized_ = False
        self.initializationCount = 0

        if filename is not None:
            self.LoadGraphAndValues(filename=str(filename), loadGPS=loadGPS)

        # Noise models
        self.odomNoise = gtsam.noiseModel.Diagonal.Variances(np.asarray([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]))
    
    def LoadGraphAndValues(self, filename:str, loadGPS:bool=True, symbolChar:str='') -> None: 
        self.orig_graph_, self.orig_values_ = fgtools.readG2oWithPrior(filename, loadGPS=loadGPS)
        self.SetGraphAndValsToOrig()
        if symbolChar:
            self.RenameKeys(symbolChar)

    def SetGraphAndValues(self, graph: gtsam.gtsam.NonlinearFactorGraph, values:gtsam.gtsam.Values, symbolChar:str='', optimize:bool=False) -> None:
        self.orig_graph_ = deepcopy(graph)
        self.orig_values_ = deepcopy(values)
        self.SetGraphAndValsToOrig()
        if symbolChar:
            self.RenameKeys(symbolChar)
        
        if optimize:
            self.UpdateISAM2()

    def SetGraphAndValsToOrig(self):
        self.graph_ = deepcopy(self.orig_graph_)
        self.values_ = deepcopy(self.orig_values_)
        self.currentEstimate_ = deepcopy(self.orig_values_)

        # self.UpdateISAM2(reset=True)
    
    def UpdateISAM2(self, reset:bool=False, removeAllCurrent:bool=False):
        if reset:
            self.isam_ = gtsam.ISAM2(params=self.isamParams_)
            # self.sessionSize = {}
            self.sessionKeys = {}

        constrained_keys = gtsam.KeyGroupMap()

        if self.params.fixedLagCurrentSession:

            for k in self.values_.keys():
            # for k in self.sessionKeys[self.currSessionSymbol_]:
                if gtsam.Symbol(k).chr() == ord(self.currSessionSymbol_):
                    constrained_keys.insert2(k, 0)

        self.IncrementSize(self.values_.keys())
        
        result = self.isam_.update(self.graph_, self.values_, [], constrained_keys)
        self.isam_.update()
        # self.currentEstimate_ = self.isam_.calculateEstimate()

        if self.params.fixedLagCurrentSession:

            new_indices = result.getNewFactorsIndices()

            for i in range(len(new_indices)):
                f = self.graph_.at(i)
                if self.IsBetweenSessionFactor(f):
                    added = False
                    for k in f.keys():
                        if gtsam.Symbol(k).chr() == ord(self.currSessionSymbol_):
                            if added:
                                raise Warning("Code issue. A factor was said to be between session but has more than one current session keys in it")
                            if k in self.betweenSessionLoopClosureIndices_:
                                self.betweenSessionLoopClosureIndices_[k].append(new_indices[i])
                            else:
                                self.betweenSessionLoopClosureIndices_[k] = [new_indices[i]]
                            added = True

        self.graph_.resize(0)
        self.values_.clear()

        if self.params.fixedLagCurrentSession:
            if self.decisionState == NodeDecision.LOCALIZE:

                keysToRemove = self.DecrementSize(fastlist=True, removeAll=removeAllCurrent)
            else:
                keysToRemove = gtsam.KeyList()

            

            if isinstance(keysToRemove, gtsam.KeyList):
                if keysToRemove.size():
                    constrained_keys = gtsam.KeyGroupMap()
                    indices_to_remove = []


                    for k in keysToRemove:
                        if k in self.betweenSessionLoopClosureIndices_:
                            indices_to_remove += self.betweenSessionLoopClosureIndices_.pop(k) # Removes the key from the dictionary

                    for k in keysToRemove:
                        constrained_keys.insert2(k, 0)

                    self.isam_.update(gtsam.NonlinearFactorGraph(), gtsam.Values(), indices_to_remove, constrained_keys)
                    self.isam_.update()
                    self.isam_.marginalizeLeaves(keysToRemove)

        self.currentEstimate_ = self.isam_.calculateEstimate()

        # if hasattr(self, "printSlidingWindow"):
        #     if self.printSlidingWindow:
        #         currGraph = fgtools.cleanFactorGraph(self.isam_)
        #         for i in range(currGraph.size()):
        #             f = currGraph.at(i)
        #             for k in f.keys():
        #                 if gtsam.Symbol(k).chr() == ord(self.currSessionSymbol_):
        #                     print(gtsam.Symbol(k))
        #                     f.print()
        #                     break

    def AddCurrentToPrior(self):

        # for k in self.sessionKeys[self.currentSessionSymbol]:
        #     if k in self.currentEstimate_.keys():
        #         return
            
        self.addMultiSessionToPrior_ = False

        currGraph = fgtools.cleanFactorGraph(self.isam_)
        indicesToPrior = []

        for v in self.betweenSessionLoopClosureIndices_.values():
            indicesToPrior += v
        
        addBetweenLastAddToPriorKey = False

        if self.addToPriorSymbol_ in self.sessionKeys.keys():

            lastAddToPriorKey = self.sessionKeys[self.addToPriorSymbol_][-1]

            if gtsam.Symbol(self.currSessionSymbol_ , gtsam.Symbol(lastAddToPriorKey).index()).key() in self.sessionKeys[self.currSessionSymbol_]:
                dontAddKeys = [gtsam.Symbol(self.currSessionSymbol_ , gtsam.Symbol(k).index()).key() for k in self.sessionKeys[self.addToPriorSymbol_]]
                keysToAdd = set(self.sessionKeys[self.currSessionSymbol_]) - set(dontAddKeys)
                addBetweenLastAddToPriorKey = True
            else:
                keysToAdd = set(self.sessionKeys[self.currSessionSymbol_])
        else:
            keysToAdd = set(self.sessionKeys[self.currSessionSymbol_])


        graph, values = fgtools.createSubGraph(currGraph, self.currentEstimate_, keysToAdd)

        for idx in indicesToPrior:
            f = currGraph.at(idx)
            if len(keysToAdd & set(f.keys())):
                graph.add(currGraph.at(idx))

        if addBetweenLastAddToPriorKey:
            keysToFindSet = set([gtsam.Symbol(self.currSessionSymbol_, gtsam.Symbol(lastAddToPriorKey).index()).key(), list(keysToAdd)[0]])
            for i in range(currGraph.size()):
                f = currGraph.at(i)
                if not (set(f.keys()) - keysToFindSet):
                    new_keys = [gtsam.Symbol(self.addToPriorSymbol_, gtsam.Symbol(k).index()).key() for k in f.keys()]
                    graph.add(f.rekey(new_keys)) 

        
        graph, values = self.RenameKeys(self.addToPriorSymbol_, graph, values, updateIsam2=True, resetIsam2=False)

        # self.graph_.push_back(graph)
        # self.values_.insert(values)
        # self.UpdateISAM2(reset=False, removeAllCurrent=True)
        _, keys = fgtools.GetPointsFromGraphEstimate(values, symbolChr=self.addToPriorSymbol_)
        self.loopKeysToIgnore |= set(keys)
        self.priorTree_ = None

    # def Optimize(self, iterations:int=1):
    #     self.isam_.update()
    #     iterations -= 1
    #     for _ in range(iterations):
    #         self.isam_.update()
    
    def UpdateDecisionState(self):
        if not self.isInitialized_:
            print("UpdateDecisionState: Waiting for initialization...")
            return
        previousState = copy(self.decisionState)
        self.UpdatePriorTree()
        self.metrics.CalculateMetrics(
            self.isam_, 
            self.currentEstimate_, 
            self.priorTree_, 
            self.sessionKeys[self.currSessionSymbol_][-1],
            self.lastInterICPResult_,
            self.pcds_
            )
        self.decisionState = self.decider.DetermineDecisionState(self.metrics)

        if previousState == NodeDecision.MAP and self.decisionState == NodeDecision.LOCALIZE:
            self.addMultiSessionToPrior_ = True
        
        if self.addMultiSessionToPrior_:
            self.AddCurrentToPrior()

        if previousState != self.decisionState:
            print(f"DecisionState: {"Localize" if self.decisionState == NodeDecision.LOCALIZE else "Map"}")

    def RenameKeys(self, 
                   chars: str|list[str]='', 
                   graph:gtsam.NonlinearFactorGraph|None=None, 
                   values:gtsam.Values|None=None,
                   autoRename:bool=False,
                   updateIsam2:bool=True, 
                   resetIsam2:bool=False):
        '''
        This function is used to rename the keys in the object graph and values or in the provided graph and values.
        If chars is left as an empty string (default), the keys will automatically be labeled in order from 'a' to 'z'

        :param chars: Either single character or list of single characters. 
        :param graph: (Optional) If not provided, will default to the object's graph
        :param values: (Optional) If not provided, will default to the object's values
        :param updateIsam2: (default=True) automatically update isam2 with the graph and values
        :param resetIsam2: (default=False) If updateIsam2 is True, this can reset the isam2 graph 
            and values and add in only the rekeyed graph and values computed in this function.
        '''

        updateSessionSymbols = False

        if graph is None or values is None:
            graph = deepcopy(self.graph)
            values = deepcopy(self.currentEstimate_)
        
        if autoRename:
            chars = [chr(i) for i in range(ord('A'), ord('A')+26)]
        elif isinstance(chars, str):
            chars = [chars]
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
        
        def rekeyLinearContainerFactorUnary(f, new_keys):

            hf = f.factor()  # HessianFactor

            info = hf.augmentedInformation()

            #
            # Split augmented info matrix
            #
            G = info[:-1, :-1]
            g = info[:-1, -1]
            c = info[-1, -1]

            #
            # Rebuild HessianFactor with new key
            #
            new_hf = gtsam.HessianFactor(
                new_keys,
                G,
                g,
                c
            )

            #
            # Rebuild LinearContainerFactor with new HessianFactor
            #
            return gtsam.LinearContainerFactor(new_hf)
        
        keyMap = {}
        for key in values.keys():
            keyMap[key] = renameKey(key)
        
        new_graph = gtsam.NonlinearFactorGraph()
        for i in range(graph.size()):
            f = graph.at(i)

            new_keys = [keyMap[k] if k in keyMap else k for k in f.keys()]
            
            # The list length MUST equal f.keys().size()
            assert len(new_keys) == len(f.keys()), f"Expected {len(f.keys())} keys, got {len(new_keys)}"

            if isinstance(f, gtsam.LinearContainerFactor):
                if len(new_keys) == 1:
                    new_f = rekeyLinearContainerFactorUnary(f, new_keys[0])
                else:
                    raise ValueError(f"Error: More than one key in LinearContainerFactor. Keys: {new_keys}")
            else:
                new_f = f.rekey(new_keys)
            new_graph.push_back(new_f)

        new_values = gtsam.Values()
        for old_key, new_key in keyMap.items():
            new_values.insert(new_key, values.atPose3(old_key))


        if self.pcds_:
            pcdKeys = list(self.pcds_.keys())
            for key in pcdKeys:
                if key in keyMap.keys():
                    if resetIsam2:
                        self.pcds_[keyMap[key]] = self.pcds_.pop(key)
                    else:
                        self.pcds_[keyMap[key]] = self.pcds_[key]
            
            if not set(new_values.keys()).issubset(self.pcds_.keys()):
                print(f"pcd keys don't match value's keys{[f" {gtsam.Symbol(k).string()}" for k in (set(new_values.keys()) - set(self.pcds_.keys()))]}")
        
        if autoRename:
            self.priorSessionSymbols_ = list(usedChars.values())
            self.addToPriorSymbol_ = chr(ord(self.priorSessionSymbols_[-1]) + 1)

            print(f"PriorSessionSymbols have been updated to {self.priorSessionSymbols_}")
            print(f"AddToPriorSymbol has been updated to {self.addToPriorSymbol_}")
        
        if updateIsam2:
            self.graph_.push_back(new_graph)
            self.values_.insert(new_values)
            self.UpdateISAM2(reset=resetIsam2)
        
        return new_graph, new_values
    
    def LoadScans(self, directory:str|Path, symbolChr:str=''):
        pcds = fgtools.loadPosePCDs(str(directory))
        if symbolChr:
            for k, v in pcds.items():
                self.pcds_[gtsam.Symbol(symbolChr, k).key()] = v
        else:
            self.pcds_ = pcds
        assert len(self.pcds_.keys()) == len(self.currentEstimate_.keys()), "Number of PCDs must match number of values in the graph. Check to make sure you loaded values first."


    def CreateGlobalPCD(self, color: tuple[float, float, float] | None = None, voxel_size: float | None = None) -> o3d.geometry.PointCloud:
        if not self.pcds_:
            raise Exception("Must load PCDs first before creating global point cloud")
        return fgtools.createGlobalPCD(self.pcds_, self.currentEstimate_, color=color, voxel_size=voxel_size)

    def AddFactor(self, factor: gtsam.gtsam.Factor, key:int|None = None, estimate: gtsam.gtsam.Pose3|None=None) -> None:
        self.graph_.add(factor)

        if key is not None and estimate is not None:
            self.values_.insert_pose3(key, estimate)

        self.UpdateISAM2()

    def AddPrior(self, key: int, pose: gtsam.Pose3, noise: gtsam.noiseModel.Base, skipValue:bool=False):
        f = gtsam.PriorFactorPose3(key, pose, noise)
        if skipValue:
            self.AddFactor(f)
        else:
            self.AddFactor(f, key, pose)
    
    def AddOdomFactor(self, pose: gtsam.Pose3, relPose:gtsam.Pose3|None=None, symbolChr:str='') -> int:
        if self.currentEstimate_.empty() or not symbolChr in self.sessionKeys:
            raise RuntimeError(f"Values is empty or values for symbol {symbolChr} is empty. Must add a prior value before adding odometry factor")
        
        if self.params.fixedLagCurrentSession:
            self.UpdateDecisionState()
        
        lastKey = self.sessionKeys[symbolChr][-1]
        if relPose is None:
            poseFrom = self.currentEstimate_.atPose3(lastKey)
            relPose = poseFrom.between(pose)

        factor = gtsam.BetweenFactorPose3(
            lastKey,
            lastKey + 1,
            relPose,
            self.odomNoise
        )

        self.AddFactor(factor, lastKey + 1, pose)

        return lastKey + 1
    
    def AddPointCloud(self, key: int, pointCloud: o3d.geometry.PointCloud) -> bool:
        if pointCloud in self.pcds_:
            return False
        
        self.pcds_[key] = pointCloud
        return True
    
    def SetPointClouds(self, scans: dict[int, o3d.geometry.PointCloud]) -> None:
        self.pcds_ = deepcopy(scans)
    
    def UpdatePriorTree(self):
        if self.priorTree_:
            return
        points, keys = fgtools.GetPointsFromGraphEstimate(self.currentEstimate_, self.currSessionSymbol_, allOthers=True)

        if not points: # If points is empty
            print("Warning: Can't perform inter session loopclosure because prior Session data points is empty. Please check symbol name provided in params")
            return
        
        if len(self.loopKeysToIgnore):
            tmpPoints, tmpKeys = [], []
            for p, k in zip(points, keys):
                if k not in self.loopKeysToIgnore:
                    tmpPoints.append(p)
                    tmpKeys.append(k)
            points, keys = tmpPoints, tmpKeys

        tree = KDTree(points)
        self.priorTree_ = TreeData(tree, keys, points)
    
    def CheckLoopClosures(self, intra:bool=True, inter:bool=True) -> None:
        '''
        Check for loop closures if specified (default=True) and optimize the graph
        '''
        # Intra loop closures (same session)
        if intra:
            if self.Size(self.currSessionSymbol_) > self.params.loopPrevFrameSkip + 1 and self.decisionState == NodeDecision.MAP:
                points, keys = fgtools.GetPointsFromGraphEstimate(self.currentEstimate_, self.currSessionSymbol_)
                points = np.asarray(points)
                self.tree_[self.currSessionSymbol_] = KDTree(points[:-self.params.loopPrevFrameSkip,:]) # Don't put last 10 poses in the tree
                distances, indices = self.tree_[self.currSessionSymbol_].query(points[-1,:], k=5, distance_upper_bound=self.params.loopDistMax) # Look for the 5 closest poses given the distance is less than 30

                keysToSearch = np.asarray(keys)[indices[distances < self.params.loopDistMax]]
                self.LoopCheckGivenKeys(keysToSearch)

        # Inter loop closures
        if inter and self.priorSessionSymbols_:
            pose = self.currentEstimate_.atPose3(self.sessionKeys[self.currSessionSymbol_][-1])
            if len(self.loopKeysToIgnore):
                tmpPoints, tmpKeys = fgtools.GetPointsFromGraphEstimate(self.currentEstimate_, keys=list(self.loopKeysToIgnore))
                tmpTree = KDTree(tmpPoints)
                tmpIndices = tmpTree.query_ball_point(pose.translation(), r=self.params.loopDistMax)
                size = len(self.loopKeysToIgnore)
                self.loopKeysToIgnore &= set(np.asarray(tmpKeys)[tmpIndices])
                if len(self.loopKeysToIgnore) != size:
                    self.priorTree_ = None
            self.UpdatePriorTree()
            # distances, indices = self.priorTree_.tree.query(pose.translation(), k=5, distance_upper_bound=self.params.loopDistMax) # Look for the 5 closest poses given the distance is less than 30
            distances, indices = self.priorTree_.tree.query(pose.translation()) # Look for the 5 closest poses given the distance is less than 30
            keys = deepcopy(self.priorTree_.keys)

            keysToSearch = np.asarray(keys)[indices[distances < self.params.loopDistMax]]
            reg = self.LoopCheckGivenKeys(keysToSearch)
            if reg:
                self.lastInterICPResult_ = reg
                if not self.isInitialized_:
                    if reg.fitness > 0.5:
                        self.initializationCount += 1
                        if self.initializationCount >= 2:
                            self.isInitialized_ = True
                            print("*** Initialized ***")
                    else:
                        self.initializationCount = 0
                    

        
    
    def LoopCheckGivenKeys(self, keysToSearch):
        if not len(keysToSearch):
            return None
        
        currKey = self.sessionKeys[self.currSessionSymbol_][-1]
        
        for key in keysToSearch:

            pcSource = self.FindNearKeyFrames(currKey, 0, -1)
            pcTarget = self.FindNearKeyFrames(key, self.params.icpNearKeyFrames, -1)

            poseTarget = self.currentEstimate_.atPose3(key)

            reg = fgtools.ScanMatch(pcSource, pcTarget, np.eye(4), showResult=False)
            fit = reg.fitness
            transFinal = reg.transformation

            if fit > self.params.icpFitnessThreshold:
                # ScanMatch(pcSource, pcTarget, np.eye(4), showResult=True)
                cs = gtsam.Symbol(currKey)
                ts = gtsam.Symbol(key)
                if self.params.debugLoop:
                    print(f"Loop Found between keys {chr(cs.chr()) if cs.chr() else ''}{cs.index()} and {chr(ts.chr()) if ts.chr() else ''}{ts.index()}")
                loopNoise = gtsam.noiseModel.Diagonal.Variances(np.asarray([(1-fit) * self.params.loopNoiseScale] * 6))

                Twrong = self.currentEstimate_.atPose3(currKey).matrix()

                Tfixed = transFinal @ Twrong
                posefixed = gtsam.Pose3(Tfixed)
                # draw_registration_result(self.pcds_[currKey], pcTarget, Tfixed, "Fixed Tranform")
                self.AddFactor(gtsam.BetweenFactorPose3(currKey, key, posefixed.between(poseTarget), loopNoise))
                break
        
        return reg

    
    def FindNearKeyFrames(self, key:int, searchNum:int, loop_index:int, downsample:bool=False) -> o3d.geometry.PointCloud:
        localMap = o3d.geometry.PointCloud()

        searchKeys = np.arange(-searchNum, searchNum+1)
        for i in searchKeys:
            keyNear = int(key + i)
            if keyNear not in self.currentEstimate_.keys():
                continue

            pcd = deepcopy(self.pcds_[int(keyNear)])
            pcd.transform(self.currentEstimate_.atPose3(int(keyNear)).matrix())
            localMap += pcd
        
        if downsample:
            localMap = localMap.voxel_down_sample(0.5) # voxelize localMap

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
        
    # def InitializeSessionKeys(self):
    #     for char in [self.currSessionSymbol_, self.addToPriorSymbol_] + self.priorSessionSymbols_:
    #         if char not in self.sessionKeys:
    #             self.sessionKeys[char] = []

    
    def IncrementSize(self, keys: int|list[int]):
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
            # self.sessionSize[char] = len(self.sessionKeys[char])
    
    def DecrementSize(self, finalSize:int|None=None, char:str|None=None, fastlist:bool=True, removeAll:bool=False) -> list[int]|gtsam.KeyList:
        if finalSize is None:
            finalSize = self.params.fixedLagKeepFrames
        if char is None:
            char = self.currSessionSymbol_

        if not char in self.sessionKeys or len(self.sessionKeys[char]) <= finalSize:
            return gtsam.KeyList()
        
        if removeAll:
            keysToRemove = self.sessionKeys[char][:-1]
            self.sessionKeys[char] = self.sessionKeys[char][-1:]
        else:
            keysToRemove = self.sessionKeys[char][:-finalSize]
            self.sessionKeys[char] = self.sessionKeys[char][-finalSize:]
            assert finalSize == len(self.sessionKeys[char]), f"Logic error in decrement size. finalSize {finalSize} != len(self.sessionKeys[char]) {len(self.sessionKeys[char])}"

        for k in keysToRemove:
            self.pcds_.pop(k)

        if fastlist:
            ret = gtsam.KeyList()
            for k in keysToRemove:
                ret.push_back(k)
            return ret
        
        return keysToRemove
    
    def IsBetweenSessionFactor(self, factor: gtsam.Factor) -> bool:
        '''
        Check to see if a factor is between the current session and a previous session
        '''
        chars = set([gtsam.Symbol(k).chr() for k in factor.keys()])

        if ord(self.currSessionSymbol_) in chars:
            return chars != {ord(self.currSessionSymbol_)} # If chars doesn't just contain the current session, then it's a between session factor
        else:
            return False
        # for k in factor.keys():
        #     s = gtsam.Symbol(k)
        #     if not firstSymbol:
        #         firstSymbol = s.chr()
        #     else:
        #         if s.chr() != firstSymbol:
        #             return True
        # return False

    def Finalize(self):
        '''
        Removed fixed lag smother section if applicable.
        If still in mapping mode, add fixed lag section to prior.
        '''
        if self.params.fixedLagCurrentSession:
            if self.decisionState == NodeDecision.MAP:
                self.AddCurrentToPrior()
        
        self.graph_ = self.currPriorGraph
        self.values_ = self.currPriorValues
        for k in self.sessionKeys[self.currSessionSymbol_]:
            self.pcds_.pop(k)
        self.UpdateISAM2(reset=True)
        

    @property
    def currPriorGraph(self):
        return fgtools.cleanFactorGraph(self.isam_, excludedSymbols=[self.currSessionSymbol_])
    
    @property
    def currPriorValues(self):
        newValues = gtsam.Values()
        for k in self.currentEstimate_.keys():
            if chr(gtsam.Symbol(k).chr()) == self.currSessionSymbol_:
                continue
            p = self.currentEstimate_.atPose3(k)
            newValues.insert(k, p)
        
        return newValues
        

    @property
    def graph(self) -> gtsam.gtsam.NonlinearFactorGraph:
        return fgtools.cleanFactorGraph(self.isam_)

    @property
    def values(self) -> gtsam.gtsam.Values:
        return self.currentEstimate_

    @property
    def scans(self) -> dict[int, o3d.geometry.PointCloud]:
        return self.pcds_
    
    @property
    def currentSessionSymbol(self) -> str:
        return deepcopy(self.currSessionSymbol_)
    
    @property
    def addToPriorSymbol(self) -> str:
        return deepcopy(self.addToPriorSymbol_)

    @property
    def priorSessionSymbols(self) -> list[str]:
        return deepcopy(self.priorSessionSymbols_)
    
    @property
    def isInitialized(self) -> bool:
        return self.isInitialized_