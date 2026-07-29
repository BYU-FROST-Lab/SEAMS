#!/usr/bin/env python3

import os

import numpy as np
np.set_printoptions(linewidth=np.inf, suppress=True)
from copy import copy, deepcopy
from pathlib import Path
import open3d as o3d
from scipy.spatial import KDTree
import time

# Local imports
import common.FGVizHelpers as fgtools
import gtsam
from params import Params
from common.customTypes import *
from metrics import Metrics
from policy.geometric import DistancePolicy
from policy.registration import FitnessPolicy
from policy.graphTheoretic import TwoStepConnectivityPolicy
from policy.probabilistic import RelativeCovariancePolicy

class SessionManager:
    def __init__(self, priorGraphFilename:str|Path|None=None, loadGPS:bool=True, params:Params=Params()):
        self.params = params
        
        self.graph_ = gtsam.NonlinearFactorGraph()
        self.values_ = gtsam.Values()

        # I have gotten weird errors where calculateEstimate() tries to find a value I've already removed and throws an error.
        # I'm not sure why, but it works with these parameters.
        self.isamParams_ = gtsam.ISAM2Params()
        self.isamParams_.setRelinearizeThreshold(0.0) # Must be zero to work
        self.isamParams_.relinearizeSkip = 1 # Must be small (<4) to work
        self.isamParams_.findUnusedFactorSlots = True # Docs say to use this when doing fixedLagSmoother type things.
        # self.isamParams_.evaluateNonlinearError = True

        self.isam_ = gtsam.ISAM2(self.isamParams_)
        self.smoother_ = gtsam.IncrementalFixedLagSmoother(float(self.params.fixedLagKeepFrames))
        self.currTimestamp_ = 0.0
        self.currentEstimate_ = gtsam.Values()

        self.pcds_: dict[int, o3d.geometry.PointCloud] = {}

        self.temporaryLoopClosureIndices_: dict[int, list[int]] = {}

        self.decisionState_ = NodeDecision.INITIALIZE
        if self.params.policyType == DecisionType.DISTANCE:
            self.decider = DistancePolicy(self.params)
        elif self.params.policyType == DecisionType.FITNESS:
            self.decider = FitnessPolicy(self.params)
        elif self.params.policyType == DecisionType.TWO_STEP_VALIDATION:
            self.decider = TwoStepConnectivityPolicy(self.params)
        elif self.params.policyType == DecisionType.RELATIVE_COVARIANCE:
            self.decider = RelativeCovariancePolicy(self.params)
        else:
            raise ValueError(f"SessionManager: Invalid decision policy chosen. {self.params.policyType}")
        
        self.metrics_ = Metrics(self.params.policyType)

        self.loopKeysToIgnore_: set[int]  = set()
        self.lastInterICPResult_ = None
        
        self.currentSessionSymbol_ = self.params.currSessionSymbolChr
        self.addToPriorSymbol_ = self.params.addToPriorSymbolChr
        self.priorSessionSymbols_ = self.params.priorSessionSymbolChrs

        self.sessionKeys: dict[str, list[int]] = {}
        self.lastOdomKey_ = 0

        self.isInitialized_ = False
        self.initializationCount_ = 0

        # Prior graph
        self.priorGraph_ = gtsam.NonlinearFactorGraph()
        self.priorValues_ = gtsam.Values()
        self.priorIsam_ = gtsam.ISAM2()

        self.currentToPriorAnchors_: dict[int, list[int]] = {}
        self.priorAnchorsToCurrent_: dict[int, list[int]] = {}
        self.priorKeyToFactorIndexDict_: dict[int, list[int]] = {}
        if priorGraphFilename is not None:
            self.LoadPriorGraphAndValues(filename=str(priorGraphFilename), loadGPS=loadGPS, symbolChar=self.params.priorSessionSymbolChrs)


        # Noise models
        self.odomNoise_ = gtsam.noiseModel.Diagonal.Variances(np.asarray([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]))
    
    def LoadPriorGraphAndValues(self, filename:str, loadGPS:bool=True, symbolChar:list[str]|None=None) -> None: 
        graph, values = fgtools.readG2oWithPrior(filename, loadGPS=loadGPS)
        self.SetPriorGraphAndValues(graph, values, symbolChar)
        
    def SetPriorGraphAndValues(self, graph: gtsam.gtsam.NonlinearFactorGraph, values:gtsam.gtsam.Values, symbolChar:list[str]|None=None) -> None:
        graph, values = self.RenameKeys(symbolChar, graph, values, autoRename=self.params.autoRename, updateIsam2=False)
        self.priorGraph_ = deepcopy(graph)
        self.priorValues_ = deepcopy(values)
        self.priorIsam_ = gtsam.ISAM2()
        self.priorIsam_.update(self.priorGraph_, self.priorValues_)
        self.priorIsam_.update()
        self.IncrementSize(self.priorValues_.keys())
        self.UpdatePriorTree()
        self.UpdatePriorKeyToFactorIndexDict()

        # Used when creating dense factor for prior map
        self.UpdateSmoother()

    def UpdateSmoother(self, reset:bool=False, newLag:float|None=None):
        # Reset if needed
        if reset:
            self.smoother_ = gtsam.IncrementalFixedLagSmoother(float(self.params.fixedLagKeepFrames))
            self.currentEstimate_ = gtsam.Values()
        
        # Update lag of smoother to match newLag argument. Need to recreate fixed lag smoother in python.
        #   In c++ you can just write to the returned reference from smootherLag()
        if newLag is not None:
            self.smoother_.setSmootherLag(newLag) # Added in custom wheel

        self.IncrementSize(self.values_.keys())
        timestamps = {}
        # if gtsam.Symbol('c', 592).key() in self.values_.keys():
        #     fgtools.show_graphviz(self.smoother_.getISAM2().dot())
        #     self.showGraphviz = True
        # elif hasattr(self, "showGraphviz"):
        #     fgtools.show_graphviz(self.smoother_.getISAM2().dot())

        if self.params.fixedLagMarg == FixedLagMarginalizationType.DENSE or self.params.fixedLagMarg == FixedLagMarginalizationType.APPROX_PRIOR:
            # Include full prior Graph in fixed lag
            for k in self.priorValues_.keys():
                timestamps[k] = self.currTimestamp_
            if self.currentEstimate_.size() == 0:
                self.smoother_.update(self.priorGraph_, self.priorValues_, timestamps)

        # If either mapping or using dense marginalization, I want to keep the nodes that are being added to the prior map
        # if (self.decisionState_ == NodeDecision.MAP or self.params.fixedLagMarg == FixedLagMarginalizationType.DENSE) and self.addToPriorSymbol_ in self.sessionKeys:
        if self.addToPriorSymbol_ in self.sessionKeys:
            for k in self.sessionKeys[self.addToPriorSymbol_]:
                timestamps[k] = self.currTimestamp_


        for k in self.values_.keys():
            timestamps[k] = self.currTimestamp_
        start = time.time()
        ret = self.smoother_.update(self.graph_, self.values_, timestamps)
        print(f"Time to optimize smoother: {time.time() - start}")
        self.graph_.resize(0)
        self.values_.clear()
        self.currentEstimate_ = self.smoother_.calculateEstimate()

        return ret

    def ApproximateLinearFactorAsPrior(self, marginalFactorsIdxs: list):
        if not marginalFactorsIdxs:
            return
        # Find the linear factors
        currGraph = self.smoother_.getFactors()
        for idx in marginalFactorsIdxs:
            f = currGraph.at(idx)

            # Factors can be null due to the way the fixed lag smoother works. If a factor is null, it means it has been marginalized out and we can skip it.
            if not f:
                continue
            
            # Don't need to worry about unary factors.
            if len(f.keys()) < 2:
                continue

            if isinstance(f, gtsam.LinearContainerFactor):
                hf = f.toHessian()
                aug_info = hf.augmentedInformation()

                G = aug_info[:-1,:-1] # Information matrix portion
                g = aug_info[:-1, -1] # Information vector portion
                c = aug_info[ -1, -1] # Constant term (not used)

                # The information vector is defined as g = G * mu, where mu is the mean of the distribution. 
                # Therefore, we can solve for mu as mu = G^-1 * g
                mu = np.linalg.inv(G) @ g

                dim = 6 # Pose3
                n = len(hf.keys())
                assert G.size == (dim*n * dim*n), f"Expected hessian to be of size {(dim*n * dim*n)} but is actually {G.size}"

                # Extract Block diagonal terms for each key in current session
                keyToInfo = {}
                for j, k in enumerate(hf.keys()):
                    symbol_chr = chr(gtsam.Symbol(k).chr())
                    if symbol_chr == self.currentSessionSymbol_:
                        keyToInfo[k] = [mu[dim*j:dim*(j+1)], 0.5 * G[dim*j:dim*(j+1), dim*j:dim*(j+1)]]

                # Add information for each key back into graph
                graph = gtsam.NonlinearFactorGraph()
                for k, info_block in keyToInfo.items():
                    new_hf = gtsam.HessianFactor(k, info_block[0], info_block[1])
                    graph.add(gtsam.LinearContainerFactor(new_hf))

                # Update smoother with the new priors and remove the old linear container factor
                self.smoother_.update(graph, gtsam.Values(), {}, [idx])
                

    def UpdateISAM2(self, reset:bool=False, removeAllCurrent:bool=False):
        if reset:
            # self.isam_ = gtsam.ISAM2(params=self.isamParams_)
            self.sessionKeys = {}

        constrained_keys = gtsam.KeyGroupMap()

        # if self.params.fixedLagCurrentSession:

        #     for k in self.values_.keys():
        #     # for k in self.sessionKeys[self.currentSessionSymbol_]:
        #         if gtsam.Symbol(k).chr() == ord(self.currentSessionSymbol_):
        #             constrained_keys.insert2(k, 0)

        self.IncrementSize(self.values_.keys())
        
        result = self.isam_.update(self.graph_, self.values_, [], constrained_keys)
        self.isam_.update()

        if self.params.fixedLagCurrentSession:

            new_indices = result.getNewFactorsIndices()

            for i in range(len(new_indices)):
                f = self.graph_.at(i)
                if self.IsTemporaryLoopClosureFactor(f, self.currentSessionSymbol_):
                    for k in f.keys():
                        if gtsam.Symbol(k).chr() == ord(self.currentSessionSymbol_):
                            if k in self.temporaryLoopClosureIndices_:
                                self.temporaryLoopClosureIndices_[k].append(new_indices[i])
                            else:
                                self.temporaryLoopClosureIndices_[k] = [new_indices[i]]

        self.graph_.resize(0)
        self.values_.clear()
        self.currentEstimate_ = self.isam_.calculateEstimate()


        if self.params.fixedLagCurrentSession:
            if self.decisionState_ == NodeDecision.LOCALIZE or self.decisionState_ == NodeDecision.INITIALIZE or removeAllCurrent:

                keysToRemove = self.DecrementSize(removeAll=removeAllCurrent)
                additionalKeysToRemove = []
                for ck in keysToRemove:
                    if ck in self.currentToPriorAnchors_:
                        connectedAnchorKeys = self.currentToPriorAnchors_.pop(ck)
                        for ak in connectedAnchorKeys:
                            currentKeysConnectedToAnchorKeys = self.priorAnchorsToCurrent_[ak]
                            assert ck in currentKeysConnectedToAnchorKeys, f"The current key being removed {gtsam.Symbol(ck).string()} must also be in the priorAnchorsToCurrent list connected to the anchorKey {gtsam.Symbol(ak).string()}"
                            currentKeysConnectedToAnchorKeys.remove(ck)
                            if not currentKeysConnectedToAnchorKeys: # If anchor is not connected to any other current nodes, remove it
                                additionalKeysToRemove.append(ak)
                keysToRemove += additionalKeysToRemove
            else:
                keysToRemove = []

            if len(keysToRemove):
                constrained_keys = gtsam.KeyGroupMap()
                indices_to_remove = []


                for k in keysToRemove:
                    if k in self.temporaryLoopClosureIndices_:
                        self.temporaryLoopClosureIndices_.pop(k) # Removes the key from the dictionary
                        # indices_to_remove += self.temporaryLoopClosureIndices_.pop(k) # Removes the key from the dictionary

                for k in self.currentEstimate_.keys():
                    if k in keysToRemove:
                        constrained_keys.insert2(k, 0)
                    else:
                        constrained_keys.insert2(k, 1)

                self.isam_.update(gtsam.NonlinearFactorGraph(), gtsam.Values(), indices_to_remove, constrained_keys)
                self.isam_.update()
                print(f"Attempting to remove keys {[gtsam.Symbol(k).string() for k in keysToRemove]}")
                if gtsam.Symbol('c', 40).key() in keysToRemove:
                    fgtools.show_graphviz(self.isam_.dot())
                self.isam_.marginalizeLeaves(fgtools.ListToGtsamKeyList(keysToRemove))
                # marginalIndices, deletedIndices = self.isam_.marginalizeLeavesWithIndices(fgtools.ListToGtsamKeyList(keysToRemove))
                # fgtools.show_graphviz(self.isam_.dot())
                # assert len(marginalIndices) == 1, "More than one LinearContainerFactor was created by marginalizeLeaves"
                    

                self.currentEstimate_ = self.isam_.calculateEstimate()

    def AddCurrentToPrior(self):

        # for k in self.sessionKeys[self.currentSessionSymbol]:
        #     if k in self.currentEstimate_.keys():
        #         return

        # currGraph = fgtools.cleanFactorGraph(self.isam_)
        currGraph = fgtools.cleanFactorGraph(self.smoother_)
        indicesToPrior = []

        # for v in self.temporaryLoopClosureIndices_.values():
        #     indicesToPrior += v
        # Check to see if any of the factors in the graph are between session and add them to the prior map
        for i in range(currGraph.size()):
            f = currGraph.at(i)
            if self.IsTemporaryLoopClosureFactor(f):
                indicesToPrior.append(i)
        
        addBetweenLastAddToPriorKey = False

        if self.addToPriorSymbol_ in self.sessionKeys.keys():

            lastAddToPriorKey = self.sessionKeys[self.addToPriorSymbol_][-1]

            if gtsam.Symbol(self.currentSessionSymbol_ , gtsam.Symbol(lastAddToPriorKey).index()).key() in self.sessionKeys[self.currentSessionSymbol_]:
                dontAddKeys = [gtsam.Symbol(self.currentSessionSymbol_ , gtsam.Symbol(k).index()).key() for k in self.sessionKeys[self.addToPriorSymbol_]]
                keysToAdd = set(self.sessionKeys[self.currentSessionSymbol_]) - set(dontAddKeys)
                addBetweenLastAddToPriorKey = True
            else:
                keysToAdd = set(self.sessionKeys[self.currentSessionSymbol_])
        else:
            keysToAdd = set(self.sessionKeys[self.currentSessionSymbol_])


        graph, values = fgtools.createSubGraph(currGraph, self.currentEstimate_, keysToAdd)

        for idx in indicesToPrior:
            f = currGraph.at(idx)
            if len(keysToAdd & set(f.keys())):
                graph.add(currGraph.at(idx))

        if addBetweenLastAddToPriorKey:
            keysToFindSet = set([gtsam.Symbol(self.currentSessionSymbol_, gtsam.Symbol(lastAddToPriorKey).index()).key(), list(keysToAdd)[0]])
            for i in range(currGraph.size()):
                f = currGraph.at(i)
                if not (set(f.keys()) - keysToFindSet):
                    new_keys = [gtsam.Symbol(self.addToPriorSymbol_, gtsam.Symbol(k).index()).key() for k in f.keys()]
                    graph.add(f.rekey(new_keys)) 

        
        graph, values = self.RenameKeys(self.addToPriorSymbol_, graph, values, updateIsam2=False, resetIsam2=False)

        self.priorGraph_.push_back(graph)
        self.priorValues_.insert(values)
        self.priorIsam_.update(graph, values)
        self.priorIsam_.update()

        # self.UpdateISAM2(reset=False, removeAllCurrent=True)
        # self.UpdateSmoother(reset=True, newLag=self.params.fixedLagKeepFrames)
        self.UpdateSmoother()

        _, keys = fgtools.GetPointsFromGraphEstimate(values, symbolChr=self.addToPriorSymbol_)
        self.loopKeysToIgnore_ |= set(keys)
        self.UpdatePriorTree()
        self.UpdatePriorKeyToFactorIndexDict()
    

    def UpdateDecisionState(self):
        if not self.isInitialized_:
            print("UpdateDecisionState: Waiting for Scan Matching initialization...")
            return
        previousState = copy(self.decisionState_)
        if self.priorTree_ is None:
            raise Exception("PriorTree_ can't be none to calculate metrics_")
        self.metrics_.CalculateMetrics(
            self.isam_, 
            self.currentEstimate_, 
            self.priorTree_, 
            self.lastOdomKey_,
            self.lastInterICPResult_,
            self.pcds_,
            self.decisionState_,
            self.sessionKeys[self.currentSessionSymbol]
            )
        self.decisionState_ = self.decider.DetermineDecisionState(self.metrics_)

        if self.decisionState_ == NodeDecision.INITIALIZE:
            print(f"Waiting for policy initialization...")
        
        if previousState != self.decisionState_:
            print(f"DecisionState: {"Localize" if self.decisionState_ == NodeDecision.LOCALIZE else ("Map" if self.decisionState_ == NodeDecision.MAP else "Initialize")}")

        # if previousState == NodeDecision.LOCALIZE and self.decisionState_ == NodeDecision.MAP:
        #     # self.UpdateISAM2(removeAllCurrent=True)
        #     # self.UpdateSmoother(newLag=np.inf)
        #     pass
        # elif previousState == NodeDecision.MAP and self.decisionState_ == NodeDecision.LOCALIZE:
        #     self.AddCurrentToPrior()

    def RenameKeys(self, 
                   chars: str|list[str]|None=None, 
                   graph:gtsam.NonlinearFactorGraph|None=None, 
                   values:gtsam.Values|None=None,
                   autoRename:bool=False,
                   updateIsam2:bool=True, 
                   resetIsam2:bool=False):
        '''
        This function is used to rename the keys in the object graph and values or in the provided graph and values.
        If chars is set to None (default), the keys will automatically be labeled in order from 'A' to 'Z'

        :param chars: Either single character, list of single characters, or None. If None, autoRename will be marked True. 
        :param graph: (Optional) If not provided, will default to the object's graph
        :param values: (Optional) If not provided, will default to the object's values
        :param autoRename: (default=False) If True, the symbol characters will be determined starting from 'A' to 'Z'
        :param updateIsam2: (default=True) automatically update isam2 with the graph and values
        :param resetIsam2: (default=False) If updateIsam2 is True, this can reset the isam2 graph 
            and values and add in only the rekeyed graph and values computed in this function.
        '''

        if graph is None or values is None:
            graph = deepcopy(self.priorGraph_)
            values = deepcopy(self.priorValues_)
        
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

            # if isinstance(f, gtsam.LinearContainerFactor):
            #     if len(new_keys) == 1:
            #         new_f = rekeyLinearContainerFactorUnary(f, new_keys[0])
            #     else:
            #         raise ValueError(f"Error: More than one key in LinearContainerFactor. Keys: {new_keys}")
            # else:
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
            # self.UpdateISAM2(reset=resetIsam2)
            self.UpdateSmoother()
        
        return new_graph, new_values
    
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
        assert len(self.pcds_.keys()) == len(self.currentEstimate_.keys())+len(self.priorValues_.keys()), "Number of PCDs must match number of values in the graph. Check to make sure you loaded values first."


    def CreateGlobalPCD(self, color: tuple[float, float, float] | None = None, voxel_size: float | None = None) -> o3d.geometry.PointCloud:
        if not self.pcds_:
            raise Exception("Must load PCDs first before creating global point cloud")
        return fgtools.createGlobalPCD(self.pcds_, self.currentEstimate_, color=color, voxel_size=voxel_size)

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

        # self.UpdateISAM2()
        self.UpdateSmoother()
    
    def AddOdomFactor(self, pose: gtsam.Pose3, relPose:gtsam.Pose3|None=None, symbolChr:str='') -> int:
        if symbolChr == '':
            if self.decisionState_ == NodeDecision.LOCALIZE:
                symbolChr = self.currentSessionSymbol_
            else:
                symbolChr = self.addToPriorSymbol_
        if self.currentEstimate_.empty():
            raise RuntimeError(f"Values is empty. Must add a prior value before adding odometry factor")
        
        if self.params.fixedLagCurrentSession:
            self.UpdateDecisionState()
        
        
        if relPose is None:
            poseFrom = self.currentEstimate_.atPose3(self.lastOdomKey_)
            relPose = poseFrom.between(pose)

        # If not mapping, the key should be the fixed lag key
        if self.decisionState_ != NodeDecision.MAP:
            if gtsam.Symbol(self.lastOdomKey_).chr() == ord(self.currentSessionSymbol_):
                newKey = self.lastOdomKey_ + 1
            else:
                if self.currentSessionSymbol_ in self.sessionKeys:
                    newKey = self.sessionKeys[self.currentSessionSymbol_][-1] + 1
                else:
                    newKey = gtsam.Symbol(self.currentSessionSymbol_, 0).key()
        # If mapping, the key will be the new session key being added to the prior
        else:
            if gtsam.Symbol(self.lastOdomKey_).chr() == ord(self.addToPriorSymbol_):
                newKey = self.lastOdomKey_ + 1
            else:
                if self.addToPriorSymbol_ in self.sessionKeys:
                    newKey = self.sessionKeys[self.addToPriorSymbol_][-1] + 1
                else:
                    newKey = gtsam.Symbol(self.addToPriorSymbol_, 0).key()
                
        factor = gtsam.BetweenFactorPose3(
            self.lastOdomKey_,
            newKey,
            relPose,
            self.odomNoise_
        )

        self.AddFactor(factor, newKey, pose)
        # self.UpdateISAM2()
        self.currTimestamp_ += 1.0
        ret = self.UpdateSmoother()

        if self.params.fixedLagMarg == FixedLagMarginalizationType.APPROX_PRIOR:
            self.ApproximateLinearFactorAsPrior(ret.getMarginalFactorIndices())
        
        if False: # For debugging, write out the pose and covariance of each new key to a file
            with open("approxLinearContainerFactorsWithPriorAddedToEachKey.txt", "a") as f:
                if not hasattr(self, "fileFirst"):
                    self.fileFirst = False
                    f.write("Pose key, Pose covariance in row major order\n")
                cov = self.smoother_.marginalCovariance(newKey)
                to_print = f"key: {gtsam.Symbol(newKey).string()} pose: {self.currentEstimate_.atPose3(newKey).matrix().flatten()} covariance: {cov.flatten()}\n"
                # print(to_print)
                f.write(f"{to_print}")
        
        self.lastOdomKey_ = newKey

        return newKey
    
    def AddPointCloud(self, key: int, pointCloud: o3d.geometry.PointCloud) -> bool:
        if pointCloud in self.pcds_:
            return False
        
        self.pcds_[key] = pointCloud
        return True
    
    def SetPointClouds(self, scans: dict[int, o3d.geometry.PointCloud]) -> None:
        self.pcds_ = deepcopy(scans)
    
    def UpdatePriorTree(self):
        
        points, keys = fgtools.GetPointsFromGraphEstimate(self.priorValues_)

        if not points: # If points is empty
            print("Warning: Can't updatePriorTree because data points is empty.")
            return
        
        if len(self.loopKeysToIgnore_):
            tmpPoints, tmpKeys = [], []
            for p, k in zip(points, keys):
                if k not in self.loopKeysToIgnore_:
                    tmpPoints.append(p)
                    tmpKeys.append(k)
            points, keys = tmpPoints, tmpKeys

        tree = KDTree(points)
        self.priorTree_ = TreeData(tree, keys, points)
    
    def UpdatePriorKeyToFactorIndexDict(self):
        self.priorKeyToFactorIndexDict_.clear()

        for i in range(self.priorGraph_.size()):
            f = self.priorGraph_.at(i)
            keys = f.keys()
            for k in keys:
                fgtools.AddValueToDictList(self.priorKeyToFactorIndexDict_, k, i)
        
    
    def CheckLoopClosures(self, intra:bool=True, inter:bool=True) -> None:
        '''
        Check for loop closures if specified (default=True) and optimize the graph
        '''
        # Intra loop closures (same session)
        if intra:
            if self.Size(self.currentSessionSymbol_) > self.params.loopClosure.prevFrameSkip + 1 and self.decisionState_ == NodeDecision.MAP:
                points, keys = fgtools.GetPointsFromGraphEstimate(self.currentEstimate_, [self.currentSessionSymbol_, self.addToPriorSymbol_])
                points = np.asarray(points)
                tree = KDTree(points[:-self.params.loopClosure.prevFrameSkip,:]) # Don't put last 10 poses in the tree
                distances, indices = tree.query(points[-1,:], k=5, distance_upper_bound=self.params.loopClosure.distMax) # Look for the 5 closest poses given the distance is less than 30
                distances, indices = tree.query(points[-1,:]) # Look for the 5 closest poses given the distance is less than 30

                keysToSearch = np.asarray(keys)[indices[distances < self.params.loopClosure.distMax]]
                for k in keysToSearch:
                    self.LoopCheckGivenKey(k)

        # Inter loop closures
        if inter and self.priorSessionSymbols_:
            pose = self.currentEstimate_.atPose3(self.lastOdomKey_)
            if len(self.loopKeysToIgnore_):
                tmpPoints, tmpKeys = fgtools.GetPointsFromGraphEstimate(self.priorValues_, keys=list(self.loopKeysToIgnore_))
                tmpTree = KDTree(tmpPoints)
                tmpIndices = tmpTree.query_ball_point(pose.translation(), r=self.params.loopClosure.distMax)
                size = len(self.loopKeysToIgnore_)
                self.loopKeysToIgnore_ &= set(np.asarray(tmpKeys)[tmpIndices])
                if len(self.loopKeysToIgnore_) != size:
                    self.UpdatePriorTree()
            distances, indices = self.priorTree_.tree.query(pose.translation(), k=self.params.loopClosure.maxPossibleConnections, distance_upper_bound=self.params.loopClosure.distMax) # Look for the 5 closest poses given the distance is less than 30
            # distances, indices = self.priorTree_.tree.query(pose.translation()) # Look for the 5 closest poses given the distance is less than 30
            keys = deepcopy(self.priorTree_.keys)

            keysToSearch = np.asarray(keys)[indices[distances < self.params.loopClosure.distMax]]
            # reg = self.LoopCheckGivenKeys(keysToSearch)
            reg = None
            num_connections = 0
            for k in keysToSearch:
                if not self.params.fixedLagMarg == FixedLagMarginalizationType.UNARY: # True for DENSE, PRIOR_ON_ANCHOR_ALL, PRIOR_ON_ANCHOR_FIRST, THROW_AWAY_BETWEEN
                    ret = self.LoopCheckGivenKey(k, addFactor=True)
                else:
                    ret = self.LoopCheckGivenKey(k, addFactor=False)
                if reg is None:
                    reg = ret
                if ret.fitness > self.params.loopClosure.fitnessThreshold:
                    if self.isInitialized_:
                        if self.params.fixedLagMarg == FixedLagMarginalizationType.PRIOR_ON_ANCHOR_ALL \
                                or self.params.fixedLagMarg == FixedLagMarginalizationType.PRIOR_ON_ANCHOR_FIRST:
                            self.AddAnchorNode(self.lastOdomKey_, k)
                        elif self.params.fixedLagMarg == FixedLagMarginalizationType.UNARY:
                            self.AddAnchorUnary(self.lastOdomKey_, k, ret)


                    num_connections += 1
                    if num_connections >= self.params.loopClosure.maxPossibleConnections:
                        break
                
            if reg:
                self.lastInterICPResult_ = reg
                if not self.isInitialized_:
                    if reg.fitness > 0.5:
                        self.initializationCount_ += 1
                        if self.initializationCount_ >= 2:
                            self.isInitialized_ = True
                            print("*** Initialized ***")
                    else:
                        self.initializationCount_ = 0
        
        # self.UpdateISAM2()
        self.UpdateSmoother()
    
    def LoopCheckGivenKey(self, key:int, addFactor:bool=True):
        
        currKey = self.lastOdomKey_

        pcSource = self.FindNearKeyFrames(currKey, 0, -1)
        pcTarget = self.FindNearKeyFrames(key, self.params.icp.nearKeyFrames, -1)

        if self.currentEstimate_.exists(key):
            poseTarget = self.currentEstimate_.atPose3(key)
        elif self.priorValues_.exists(key):
            poseTarget = self.priorValues_.atPose3(key)
        else:
            raise Exception(f"Key {gtsam.Symbol(key).string()} is not found in currentEstimate of priorValues")

        reg = fgtools.ScanMatch(pcSource, pcTarget, np.eye(4), showResult=False)

        fit = reg.fitness
        transFinal = reg.transformation

        if fit > self.params.loopClosure.fitnessThreshold:
            cs = gtsam.Symbol(currKey)
            ts = gtsam.Symbol(key)
            if self.params.loopClosure.debug:
                print(f"Loop Found between keys {chr(cs.chr()) if cs.chr() else ''}{cs.index()} and {chr(ts.chr()) if ts.chr() else ''}{ts.index()}")
            loopNoise = gtsam.noiseModel.Diagonal.Variances(np.asarray([(1-fit) * self.params.loopClosure.noiseScale] * 6))

            Twrong = self.currentEstimate_.atPose3(currKey).matrix()

            Tfixed = transFinal @ Twrong
            posefixed = gtsam.Pose3(Tfixed)
            # draw_registration_result(self.pcds_[currKey], pcTarget, Tfixed, "Fixed Tranform")
            if self.isInitialized_ and addFactor:
                self.AddFactor(gtsam.BetweenFactorPose3(currKey, key, posefixed.between(poseTarget), loopNoise))

        return reg
    
    def AddAnchorUnary(self, currKey:int, anchorKey:int, icpRegInfo):

        # Calculate mean for prior
        transFinal = icpRegInfo.transformation

        Twrong = self.currentEstimate_.atPose3(currKey).matrix()

        Tfixed = transFinal @ Twrong
        unaryMean = gtsam.Pose3(Tfixed)

        # Scale covariance 

        scalingFactor = 2 - icpRegInfo.fitness # Puts scaling factor from [1, 2]
        # scalingFactor = 1
        # np.asarray([(1-fit) * self.params.loopNoiseScale] * 6)

        anchorCov = self.priorIsam_.marginalCovariance(anchorKey)

        p_i = self.currentEstimate_.atPose3(currKey)
        p_j = self.priorValues_.atPose3(anchorKey)

        H_i = np.zeros((6, 6), order="F")
        H_j = np.zeros((6, 6), order="F")
        p_i.between(p_j, H_i, H_j)

        
        # (scalingFactor * np.eye(anchorCov.shape[0]))
        unaryCov = H_i @ (np.diag(np.asarray([(1-icpRegInfo.fitness) * self.params.loopClosure.noiseScale] * 6)))  @ H_i.transpose() + H_j @ anchorCov @ H_j.transpose()
        noise = gtsam.noiseModel.Gaussian.Covariance(unaryCov)

        # # Test a conservative noise model TODO: Remove
        # sigma_xyz = 10.0      # meters
        # sigma_rpy = np.deg2rad(100)

        # noise = gtsam.noiseModel.Diagonal.Sigmas(
        #     np.array([
        #         sigma_rpy,
        #         sigma_rpy,
        #         sigma_rpy,
        #         sigma_xyz,
        #         sigma_xyz,
        #         sigma_xyz
        #     ])
        # )

        self.AddPrior(currKey, unaryMean, noise, skipValue=True)  


    def AddAnchorNode(self, currKey:int, anchorKey:int)->None:


        # Add any factors in the prior graph that are between this new anchor and any other current anchors
        # if anchorKey in self.priorKeyToFactorIndexDict_:
        addAnchor = anchorKey not in self.priorAnchorsToCurrent_
        addPrior = True

        fgtools.AddValueToDictList(self.currentToPriorAnchors_, currKey, anchorKey)
        fgtools.AddValueToDictList(self.priorAnchorsToCurrent_, anchorKey, currKey)

        indices = self.priorKeyToFactorIndexDict_[anchorKey]
        for i in indices:
            f = self.priorGraph_.at(i)
            if not (set(f.keys()) - set(self.priorAnchorsToCurrent_.keys())): # All keys in the factor are anchors
                self.graph_.add(f)
                if self.params.fixedLagMarg == FixedLagMarginalizationType.PRIOR_ON_ANCHOR_FIRST:
                    addPrior=False
        if addAnchor:
            # Add in a prior factor from the prior graph to the current graph. 
            # Mean is estimated value. Variance is estimated covariance.
            pose = self.priorValues_.atPose3(anchorKey)
            if addPrior:
                noise = gtsam.noiseModel.Gaussian.Information(self.priorIsam_.marginalInformation(anchorKey))
                # noise = gtsam.noiseModel.Constrained.Information(self.priorIsam_.marginalInformation(anchorKey))
                self.AddPrior(anchorKey, pose, noise)
            else:
                self.values_.insert(anchorKey, pose)
        else:
            timestamp = {anchorKey: self.currTimestamp_}
            self.smoother_.update(gtsam.NonlinearFactorGraph(), gtsam.Values(), timestamp)
        # else:
        #     print(f"Warning!!! Anchor key {gtsam.Symbol(anchorKey).string()} isn't found in priorKeyToFactorIndexDict.")


    
    def FindNearKeyFrames(self, key:int, searchNum:int, loop_index:int, downsample:bool=False) -> o3d.geometry.PointCloud:
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
    #     for char in [self.currentSessionSymbol_, self.addToPriorSymbol_] + self.priorSessionSymbols_:
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
    
    def DecrementSize(self, finalSize:int|None=None, char:str|None=None, removeAll:bool=False) -> list[int]:
        if finalSize is None:
            finalSize = self.params.fixedLagKeepFrames
        if char is None:
            char = self.currentSessionSymbol_

        if  not char in self.sessionKeys or (not removeAll and len(self.sessionKeys[char]) <= finalSize):
            tmpList = []
            return tmpList
        
        if removeAll:
            keysToRemove = self.sessionKeys[char][:-1]
            self.sessionKeys[char] = self.sessionKeys[char][-1:]
        else:
            keysToRemove = self.sessionKeys[char][:-finalSize]
            self.sessionKeys[char] = self.sessionKeys[char][-finalSize:]
            assert finalSize == len(self.sessionKeys[char]), f"Logic error in decrement size. finalSize {finalSize} != len(self.sessionKeys[char]) {len(self.sessionKeys[char])}"

        for k in keysToRemove:
            self.pcds_.pop(k)
        
        return keysToRemove
    
    def IsTemporaryLoopClosureFactor(self, factor: gtsam.Factor, symbol:str|None=None) -> bool:
        '''
        Check to see if a factor is between the current session and a previous session
        '''
        if symbol is None:
            symbolInt = ord(self.addToPriorSymbol_)
        else:
            symbolInt = ord(symbol)
            
        if len(factor.keys()) < 2:
            return False
        elif len(factor.keys()) == 2:
            s1 = gtsam.Symbol(factor.keys()[0])
            s2 = gtsam.Symbol(factor.keys()[1])

            if s1.chr() == symbolInt or s2.chr() == symbolInt:
                if s1.chr() != s2.chr():
                        return True
                else:
                    return abs(s1.index() - s2.index()) != 1
            return False
        else:
            s_list = []
            for k in factor.keys():
                s = gtsam.Symbol(k).chr()
                s_list.append(s)
            
            if symbolInt in s_list: # Has current key in it
                if set(s_list) - set([symbolInt]): # Has at least one other type of key in it
                    return True
            return False

    def Finalize(self):
        '''
        Removed fixed lag smother section if applicable.
        If still in mapping mode, add fixed lag section to prior.
        '''
        if self.params.fixedLagCurrentSession:
            if self.decisionState_ == NodeDecision.MAP:
                self.AddCurrentToPrior()
        
        self.graph_ = self.priorGraph
        self.values_ = self.priorValues
        for k in self.sessionKeys[self.currentSessionSymbol_]:
            self.pcds_.pop(k)
        # self.UpdateISAM2(reset=True)
        self.UpdateSmoother()
    


        

    @property
    def priorGraph(self):
        return self.priorGraph_
    
    @property
    def priorValues(self):
        return self.priorValues_
    
    @property
    def priorISAM2(self):
        return self.priorIsam_
    
    @property
    def graph(self):
        # return fgtools.cleanFactorGraph(self.isam_)
        return fgtools.cleanFactorGraph(self.smoother_)
        
    @property
    def values(self):
        # return self.isam_.calculateEstimate()
        return self.smoother_.calculateEstimate()
    
    @property
    def fixedLagSmoother(self):
        return self.smoother_

    @property
    def fullGraph(self) -> gtsam.gtsam.NonlinearFactorGraph:
        tmpGraph = gtsam.NonlinearFactorGraph()
        tmpGraph.push_back(self.priorGraph_)
        # tmpGraph.push_back(fgtools.cleanFactorGraph(self.isam_))4
        tmpGraph.push_back(fgtools.cleanFactorGraph(self.smoother_))
        return tmpGraph

    @property
    def fullValues(self) -> gtsam.gtsam.Values:
        tmpValues = gtsam.Values()
        tmpValues.insert(self.priorValues_)

        for k in self.currentEstimate_.keys():
            if tmpValues.exists(k):
                continue
            tmpValues.insert(k, self.currentEstimate_.atPose3(k))

        return tmpValues

    @property
    def scans(self) -> dict[int, o3d.geometry.PointCloud]:
        return self.pcds_
    
    @property
    def currentSessionSymbol(self) -> str:
        return deepcopy(self.currentSessionSymbol_)
    
    @property
    def addToPriorSymbol(self) -> str:
        return deepcopy(self.addToPriorSymbol_)

    @property
    def priorSessionSymbols(self) -> list[str]:
        # if self.priorSessionSymbols_ is None:
        #     return []
        return deepcopy(self.priorSessionSymbols_)
    
    @property
    def isInitialized(self) -> bool:
        return self.isInitialized_