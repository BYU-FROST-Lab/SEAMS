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

# Local imports
import common.FGVizHelpers as fgtools
import gtsam
import gtsam_unstable
from params import Params
from common.customTypes import *
from metrics import Metrics
from policy.base import NoDecisionPolicy
from policy.geometric import DistancePolicy
from policy.registration import FitnessPolicy, InlierRMSEPolicy
from policy.graphTheoretic import TwoStepConnectivityPolicy
from policy.probabilistic import RelativeCovariancePolicy

class SessionManager:
    def __init__(self, priorSessionDir:str|Path|None=None, params:Params=Params()):
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
        self.timestamps_: dict[int, float] = {}

        self.temporaryLoopClosureIndices_: dict[int, list[int]] = {}
        self.loopCnt_ = 0

        if self.params.fixedLagCurrentSession:
            self.decisionState_ = NodeDecision.INITIALIZE
        else:
            self.decisionState_ = NodeDecision.MAP

        if self.params.policyType == DecisionType.NONE:
            self.decider = NoDecisionPolicy(self.params)
        elif self.params.policyType == DecisionType.DISTANCE:
            self.decider = DistancePolicy(self.params)
        elif self.params.policyType == DecisionType.FITNESS:
            self.decider = FitnessPolicy(self.params)
        elif self.params.policyType == DecisionType.INLIER_RMSE:
            self.decider = InlierRMSEPolicy(self.params)
        elif self.params.policyType == DecisionType.TWO_STEP_VALIDATION:
            self.decider = TwoStepConnectivityPolicy(self.params)
        elif self.params.policyType == DecisionType.RELATIVE_COVARIANCE:
            self.decider = RelativeCovariancePolicy(self.params)
        else:
            raise ValueError(f"SessionManager: Invalid decision policy chosen. {self.params.policyType}")
        
        self.metrics_ = Metrics(self.params.policyType)
        self.mappingStartKeys_: set[int] = set()
        self.mappingEndKeys_: set[int] = set()

        self.loopKeysToIgnore_: set[int]  = set()
        self.lastInterICPResult_ = None
        self.recentKeysToSkip_: deque[int] = deque(maxlen=self.params.loopClosure.prevFrameSkip + 1)
        
        self.currentSessionSymbol_ = self.params.currSessionSymbolChr
        if self.params.fixedLagCurrentSession:
            self.addToPriorSymbol_ = self.params.addToPriorSymbolChr
        else:
            self.addToPriorSymbol_ = self.params.currSessionSymbolChr
        self.priorSessionSymbols_ = self.params.priorSessionSymbolChrs

        self.sessionKeys: dict[str, list[int]] = {}
        self.lastOdomKey_ = 0

        self.initializationCount_ = 0


        # Prior graph
        self.priorGraph_ = gtsam.NonlinearFactorGraph()
        self.priorValues_ = gtsam.Values()
        self.priorIsam_ = gtsam.ISAM2()

        self.currentToPriorAnchors_: dict[int, list[int]] = {}
        self.priorAnchorsToCurrent_: dict[int, list[int]] = {}
        self.priorKeyToFactorIndexDict_: dict[int, list[int]] = {}
        if priorSessionDir is not None:
            self.isInitialized_ = False
            self.LoadSession(priorSessionDir)
        else:
            self.isInitialized_ = True

        # Noise models
        self.odomNoise_ = gtsam.noiseModel.Diagonal.Variances(np.asarray([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]))

        self.updateInitialized_ = False
    
    def LoadPriorGraphAndValues(self, filename:str, loadGPS:bool=True, symbolChar:list[str]|None=None) -> None: 
        graph, values = fgtools.readG2oWithPrior(filename, loadGPS=loadGPS)
        self.SetPriorGraphAndValues(graph, values, symbolChar)
        
    def SetPriorGraphAndValues(self, graph: gtsam.gtsam.NonlinearFactorGraph, values:gtsam.gtsam.Values, symbolChar:list[str]|None=None) -> None:

        graph, values = self.RenameKeys(symbolChar, graph, values, autoRename=self.params.autoRename, updateOptimization=False)
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
        self.isInitialized_ = False

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

        # Include full prior Graph in fixed lag
        if self.currentEstimate_.size() == 0:
            for k in self.priorValues_.keys():
                timestamps[k] = self.currTimestamp_
            self.smoother_.update(self.priorGraph_, self.priorValues_, timestamps, [], gtsam.KeySet(self.priorValues_.keys()))

        # # Keep the nodes that are being added to the prior map
        # if self.addToPriorSymbol_ in self.sessionKeys:
        #     for k in self.sessionKeys[self.addToPriorSymbol_]:
        #         timestamps[k] = self.currTimestamp_

        keysToFreeze = gtsam.KeySet()
        for k in self.values_.keys():
            timestamps[k] = self.currTimestamp_
            if self.IsAddToPriorSessionKey(k):
                keysToFreeze.insert(k)

        # start = time.time()
        # if self.graph_.size() > 0:
        #     print(f"Adding graph:")
        #     self.graph_.print()
        #     print(f"Adding values:")
        #     self.values_.print()
        result = self.smoother_.update(self.graph_, self.values_, timestamps, [], keysToFreeze)
        # self.smoother_.update(gtsam.NonlinearFactorGraph(), gtsam.Values(), {})
        # print(f"Time to optimize smoother: {time.time() - start}")
        self.graph_.resize(0)
        self.values_.clear()
        try:
            self.currentEstimate_ = self.smoother_.calculateEstimate()

        except Exception as e:
            print(f"Error occurred while calculating estimate: {e}")
            pass

        return result

    def ApproximateLinearFactorAsPrior(self, marginalFactorsIdxs: list, maxNodesToKeep: int=1):
        if maxNodesToKeep < 1:
            raise ValueError(f"maxNodesToKeep must be greater than 0, but got {maxNodesToKeep}")
        if not marginalFactorsIdxs or not self.currentToPriorAnchors_:
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
                # key_chrs = [chr(gtsam.Symbol(k).chr()) for k in keys]

                # # Continue if doesn't contain any current session keys
                # if not self.ContainsCurrentSessionKey(f.keys()):
                #     continue

                # # Continue if any of the keys are in the mappingStartKeys_
                # if any(item in self.mappingStartKeys_ for item in keys) and not self.currentSessionSymbol_ in key_chrs:
                #     continue
                # # print([gtsam.Symbol(k).string() for k i n hf.keys()])

                # # 
                # if not self.params.currSessionSymbolChr in key_chrs and not self.params.addToPriorSymbolChr in key_chrs:
                #     continue
                
                aug_info = hf.augmentedInformation()

                G = aug_info[:-1,:-1] # Information matrix portion
                g = aug_info[:-1, -1] # Information vector portion
                c = aug_info[ -1, -1] # Constant term (not used)

                dim = 6 # Pose3
                n = len(keys)
                assert G.size == (dim*n * dim*n), f"Expected hessian to be of size {(dim*n * dim*n)} but is actually {G.size}"

                # Extract Block diagonal terms for each key in current session
                # assert self.IsCurrentSessionKey(keys[-1]), f"Expected last key in HessianFactor to be of current session symbol '{self.currentSessionSymbol_}' but is actually '{chr(gtsam.Symbol(keys[-1]).chr())}'"

                # start = time.time()

                ####################################################################################
                # Approximation using the Schur complement to marginalize out the prior graph nodes.
                ####################################################################################

                # Determine which keys to keep based on the mutual information between the prior and current session keys.
                # current_key_idxs = []
                # prior_key_idxs = []
                # for i, k in enumerate(keys):
                #     if self.IsCurrentSessionKey(k):
                #         current_key_idxs.append(i)
                #     else:
                #         prior_key_idxs.append(i)

                # assert len(current_key_idxs) == 1, f"Expected only one current session key in the HessianFactor but found {len(current_key_idxs)}"

                # mutual_info = []
                # for i in prior_key_idxs:
                #     Lambda_ii = G[dim*i:dim*(i+1), dim*i:dim*(i+1)]
                #     Lambda_ic = G[dim*i:dim*(i+1), dim*current_key_idxs[0]:dim*(current_key_idxs[0]+1)]
                #     Lambda_cc = G[dim*current_key_idxs[0]:dim*(current_key_idxs[0]+1), dim*current_key_idxs[0]:dim*(current_key_idxs[0]+1)]
                #     mutual_info.append(np.linalg.det(Lambda_ii - Lambda_ic @ np.linalg.inv(Lambda_cc) @ Lambda_ic.T) / np.linalg.det(Lambda_ii))

                # max_mutual_info_idx = np.argmax(mutual_info)
                # current_key_idxs.append(prior_key_idxs[max_mutual_info_idx]) # Keep the prior key with the highest mutual information with the current session key
                # prior_key_idxs.remove(prior_key_idxs[max_mutual_info_idx]) # Remove the prior key with the highest mutual information from the list of prior keys to marginalize out

                # Determine which keys to keep based on closest distance between the prior and current session keys.

                remove_key_idxs = []
                points, _ = fgtools.GetPointsFromGraphEstimate(self.currentEstimate_, keys=keys)
                points = np.array(points)
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
                ####################################################################################
                # Approximate by dropping information not related to the current session
                ####################################################################################
                # keyToInfo = {}

                # # The information vector is defined as g = G * mu, where mu is the mean of the distribution. 
                # # Therefore, we can solve for mu as mu = G^-1 * g
                # Sigma = np.linalg.inv(G)
                # mu = Sigma @ g

                # for j, k in enumerate(hf.keys()):
                #     symbol_chr = chr(gtsam.Symbol(k).chr())
                #     if symbol_chr == self.currentSessionSymbol_:# or symbol_chr == self.addToPriorSymbol_:
                #         keyToInfo[k] = [mu[dim*j:dim*(j+1)], Sigma[dim*j:dim*(j+1), dim*j:dim*(j+1)]]

                # # Add information for each key back into graph
                # graph = gtsam.NonlinearFactorGraph()
                # for k, info_block in keyToInfo.items():
                #     new_hf = gtsam.HessianFactor(k, info_block[0], info_block[1])
                #     graph.add(gtsam.LinearContainerFactor(new_hf))

                ####################################################################################

                # Update smoother with the new priors and remove the old linear container factor
                # end = time.time() - start
                # with open("approximationTime.txt", 'a') as file:
                #     file.write(f"{end}\n")
                # old_graph = self.smoother_.getFactors()
                # try:
                self.smoother_.update(graph, gtsam.Values(), {}, [idx])
                # except Exception as e:
                #     print(f"Error occurred while updating smoother with new prior: {e}")
                #     # old_f = old_graph.at(idx)
                #     raise Exception()
                    # pass
                # res = self.smoother_.getISAM2Result()
                # if len(newIndices) == 1:
                #     self.priorIdx_ = newIndices[0]
                # else:
                #     raise ValueError(f"Expected 1 new factor to be added to the smoother, but got {len(newIndices)}")
                

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
                        if self.IsCurrentSessionKey(k):
                            if k in self.temporaryLoopClosureIndices_:
                                self.temporaryLoopClosureIndices_[k].append(new_indices[i])
                            else:
                                self.temporaryLoopClosureIndices_[k] = [new_indices[i]]

        self.graph_.resize(0)
        self.values_.clear()
        self.currentEstimate_ = self.isam_.calculateEstimate()


        if self.params.fixedLagCurrentSession:
            if self.decisionState_ == NodeDecision.LOCALIZE or self.decisionState_ == NodeDecision.INITIALIZE or removeAllCurrent:

                keysToRemove = self.DecrementSize_legacy(removeAll=removeAllCurrent)
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
    

    def UpdateDecisionState(self) -> bool:
        '''
        This function updates the decision state based on the current metrics and the decider policy.
        
        :return: True if the decision state has changed, False otherwise'''
        if not self.isInitialized_:
            print("UpdateDecisionState: Waiting for Scan Matching initialization...")
            return False
        ret = False
        previousState = copy(self.decisionState_)
        if self.priorTree_ is None:
            raise Exception("PriorTree_ can't be none to calculate metrics_")
        self.metrics_.CalculateMetrics(
            self.smoother_.getISAM2(), 
            self.currentEstimate_, 
            self.priorTree_, 
            self.lastOdomKey_,
            self.lastInterICPResult_,
            self.pcds_,
            self.decisionState_,
            self.sessionKeys[self.currentSessionSymbol_] if self.currentSessionSymbol_ in self.sessionKeys else []
            )
        self.decisionState_ = self.decider.DetermineDecisionState(self.metrics_)

        if self.decisionState_ == NodeDecision.INITIALIZE:
            print(f"Waiting for policy initialization...")
        
        if previousState != self.decisionState_:
            print(f"DecisionState: {"Localize" if self.decisionState_ == NodeDecision.LOCALIZE else ("Map" if self.decisionState_ == NodeDecision.MAP else "Initialize")}")

        if previousState == NodeDecision.LOCALIZE and self.decisionState_ == NodeDecision.MAP:
            ret = True
            # if self.params.policyType == DecisionType.RELATIVE_COVARIANCE:
            #     self.RenameKeysInPlace({self.currentSessionSymbol_: self.addToPriorSymbol_})
            pass
        elif previousState == NodeDecision.MAP and self.decisionState_ != NodeDecision.MAP:
            ret = True
        #     self.AddCurrentToPrior()

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
                    if resetOptimization:
                        self.pcds_[keyMap[key]] = self.pcds_.pop(key)
                        self.timestamps_[keyMap[key]] = self.timestamps_.pop(key)
                    else:
                        self.pcds_[keyMap[key]] = self.pcds_[key]
                        self.timestamps_[keyMap[key]] = self.timestamps_[key]
            
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
            # self.UpdateISAM2(reset=resetIsam2)
            self.UpdateSmoother(reset=resetOptimization)
        
        return new_graph, new_values
    
    def RenameKeysInPlace(self, charMap: dict[str, str]):
        
        graph = self.smoother_.getFactors()
        timeMap = self.smoother_.timestamps()
        
        # Create key map and new Values object with rekeyed values
        keyMap: dict[int, int] = {}
        newValues = gtsam.Values()
        for k in self.currentEstimate_.keys():
            s = gtsam.Symbol(k)
            old_chr = chr(s.chr())

            if old_chr in charMap.keys():
                # Create new key with the new character and the next available index for that character
                if charMap[old_chr] not in self.sessionKeys:
                    newKey = gtsam.Symbol(charMap[old_chr], 0).key()
                else:
                    newKey = self.sessionKeys[charMap[old_chr]][-1] + 1
                

                # Update keyMap, timeMap, and newValues with the rekeyed value
                keyMap[k] = newKey
                timeMap[newKey] = timeMap[k]
                timeMap[k] = 0.0 # Reset old key's timestamp to 0.0 so that it gets removed from the smoother
                newValues.insert(newKey, self.currentEstimate_.atPose3(k))

                # Remove old key from sessionKeys and pcds
                if old_chr in self.sessionKeys:
                    self.DecrementSize(k) # Remove old
                    self.IncrementSize(newKey) # Add new
                if k in self.pcds_:
                    self.pcds_[newKey] = self.pcds_.pop(k)
                    self.timestamps_[newKey] = self.timestamps_.pop(k)

        # Create new graph with rekeyed factors
        rekeyedFactors = gtsam.NonlinearFactorGraph()
        factorIndicesToRemove = []
        for i in range(graph.size()):
            f = graph.at(i)
            if not f:
                continue
            
            # Check if any of the factor's keys are in the charMap, and if so, rekey the factor and add it to the new graph
            for k in f.keys():
                if chr(gtsam.Symbol(k).chr()) in charMap.keys():
                    new_keys = [keyMap[k] if k in keyMap else k for k in f.keys()]
                    rekeyedFactors.add(f.rekey(new_keys))
                    factorIndicesToRemove.append(i)
                    break
        
        # Update smoother with new graph and values, and remove old factors
        self.smoother_.update(rekeyedFactors, newValues, timeMap, factorIndicesToRemove)
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
            self.recentKeysToSkip_.append(key)


        # self.UpdateISAM2()
        self.UpdateSmoother()


    
    def AddOdomFactor(self, pose: gtsam.Pose3|None=None, relPose:gtsam.Pose3|None=None, symbolChr:str='') -> int:
        # if symbolChr == '':
        #     if self.decisionState_ == NodeDecision.LOCALIZE:
        #         symbolChr = self.currentSessionSymbol_
        #     else:
        #         symbolChr = self.addToPriorSymbol_
        if self.currentEstimate_.empty():
            raise RuntimeError(f"Values is empty. Must add a prior value before adding odometry factor")
        
        if self.params.fixedLagCurrentSession:
            decisionStateChanged = self.UpdateDecisionState()
        else:
            decisionStateChanged = False
        
        if pose is None and relPose is None:
            raise ValueError("AddOdomFactor: Either pose or relPose must be provided")
        
        elif pose is None and relPose is not None:
            poseFrom = self.currentEstimate_.atPose3(self.lastOdomKey_)
            pose = poseFrom.compose(relPose)
        elif relPose is None and pose is not None:
            poseFrom = self.currentEstimate_.atPose3(self.lastOdomKey_)
            relPose = poseFrom.between(pose)

        # If not mapping, the key should be the fixed lag key
        if self.decisionState_ != NodeDecision.MAP:
            if self.IsCurrentSessionKey(self.lastOdomKey_):
                newKey = self.lastOdomKey_ + 1
            else:
                newKey = gtsam.Symbol(self.currentSessionSymbol_, 0).key()
                if self.currentSessionSymbol_ in self.sessionKeys:
                    if len(self.sessionKeys[self.currentSessionSymbol_]) > 0:
                        newKey = self.sessionKeys[self.currentSessionSymbol_][-1] + 1
        # If mapping, the key will be the new session key being added to the prior
        else:
            if self.IsAddToPriorSessionKey(self.lastOdomKey_):
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
        if self.params.fixedLagCurrentSession:
            self.currTimestamp_ += 1.0
        result = self.UpdateSmoother()

        deletedKeys = result.getKeysOfDeletedNodes()
        if deletedKeys:
            for k in deletedKeys:
                self.sessionKeys[chr(gtsam.Symbol(k).chr())].remove(k)
                if k in self.currentToPriorAnchors_:
                    self.currentToPriorAnchors_.pop(k)

        if self.params.fixedLagMarg == FixedLagMarginalizationType.APPROX_PRIOR:
            self.ApproximateLinearFactorAsPrior(result.getMarginalFactorIndices())
        
        if False: # For debugging, write out the pose and covariance of each new key to a file
            with open("approxMargWithSchurComp.txt", "a") as file:
                if not hasattr(self, "fileFirst"):
                    self.fileFirst = False
                    file.write("Prior Pose key, Pose covariance in row major order\n")
                cov = self.smoother_.marginalCovariance(newKey)
                to_print = f"key: {gtsam.Symbol(newKey).string()} pose: {self.currentEstimate_.atPose3(newKey).matrix().flatten()} covariance: {cov.flatten()}\n"
                # print(to_print)
                file.write(f"{to_print}")
                # if hasattr(self, "priorIdx_"):
                #     f = self.smoother_.getFactors().at(self.priorIdx_)
                #     if isinstance(f, gtsam.LinearContainerFactor):
                #         info = f.toHessian().information()
                #         cov = np.linalg.inv(info)
                #         to_print = f"key: {gtsam.Symbol(f.keys()[0]).string()} pose: {self.currentEstimate_.atPose3(f.keys()[0]).matrix().flatten()} covariance: {cov.flatten()}\n"
                #         print(to_print)
                #         file.write(f"{to_print}")
        
        if decisionStateChanged:
            if self.decisionState_ == NodeDecision.MAP:
                self.mappingStartKeys_.add(newKey)
            else:
                assert self.IsAddToPriorSessionKey(self.lastOdomKey_), f"Expected lastOdomKey_ to be an addToPriorSessionKey but is actually {gtsam.Symbol(self.lastOdomKey_).string()}"
                self.mappingEndKeys_.add(self.lastOdomKey_)
        self.recentKeysToSkip_.append(newKey)
        self.lastOdomKey_ = newKey

        return newKey
    
    def AddPointCloud(self, key: int, pointCloud: o3d.geometry.PointCloud|np.ndarray, timestamp: float) -> bool:
        # if pointCloud in self.pcds_:
        if key in self.pcds_.keys():
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
        # self.UpdateSmoother()
        currPose = self.currentEstimate_.atPose3(self.lastOdomKey_)
        # Intra loop closures (same session)
        if intra:
            # if self.Size(self.currentSessionSymbol_) > self.params.loopClosure.prevFrameSkip + 1 and self.decisionState_ == NodeDecision.MAP:
            if self.decisionState_ == NodeDecision.MAP:
                points, keys = fgtools.GetPointsFromGraphEstimate(self.currentEstimate_, 
                                                                  [self.currentSessionSymbol_, self.addToPriorSymbol_], 
                                                                  keysToSkip=set(self.recentKeysToSkip_))
                if points:
                    points = np.asarray(points)
                    tree = KDTree(points)
                    # distances, indices = tree.query(currPose.translation(), k=5, distance_upper_bound=self.params.loopClosure.distMax) # Look for the 5 closest poses given the distance is less than 30
                    distances, indices = tree.query(currPose.translation()) # Look for the 5 closest poses given the distance is less than 30

                    keysToSearch = np.asarray(keys)[indices[distances < self.params.loopClosure.distMax]]
                    for k in keysToSearch:
                        self.LoopCheckGivenKey(k, numNearKeyFrames=self.params.loopClosure.intraNearKeyFrames)

        # Inter loop closures
        if inter and self.priorSessionSymbols_:
            
            if len(self.loopKeysToIgnore_):
                tmpPoints, tmpKeys = fgtools.GetPointsFromGraphEstimate(self.priorValues_, keys=list(self.loopKeysToIgnore_))
                tmpTree = KDTree(tmpPoints)
                tmpIndices = tmpTree.query_ball_point(currPose.translation(), r=self.params.loopClosure.distMax)
                size = len(self.loopKeysToIgnore_)
                self.loopKeysToIgnore_ &= set(np.asarray(tmpKeys)[tmpIndices])
                if len(self.loopKeysToIgnore_) != size:
                    self.UpdatePriorTree()
            distances, indices = self.priorTree_.tree.query(currPose.translation(), k=self.params.loopClosure.maxPossibleConnections, distance_upper_bound=self.params.loopClosure.distMax) # Look for the 5 closest poses given the distance is less than 30
            # distances, indices = self.priorTree_.tree.query(pose.translation()) # Look for the 5 closest poses given the distance is less than 30
            keys = deepcopy(self.priorTree_.keys)

            keysToSearch = np.asarray(keys)[indices[distances < self.params.loopClosure.distMax]]
            # reg = self.LoopCheckGivenKeys(keysToSearch)
            reg = None
            num_connections = 0
            for k in keysToSearch:
                # ret = self.LoopCheckGivenKey(k, addFactor=True, numNearKeyFrames=self.params.icp.nearKeyFrames)
                ret = self.LoopCheckGivenKey(k, addFactor=True, numNearKeyFrames=self.params.loopClosure.interNearKeyFrames)
                if reg is None:
                    reg = ret
                if ret.fitness > self.params.loopClosure.fitnessThreshold:
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
    
    def LoopCheckGivenKey(self, key:int, addFactor:bool=True, numNearKeyFrames: int = 0):
        
        currKey = self.lastOdomKey_

        pcSource = self.FindNearKeyFrames(currKey, 0, -1)
        pcTarget = self.FindNearKeyFrames(key, numNearKeyFrames, -1)

        if self.currentEstimate_.exists(key):
            poseTarget = self.currentEstimate_.atPose3(key)
        elif self.priorValues_.exists(key):
            poseTarget = self.priorValues_.atPose3(key)
        else:
            raise Exception(f"Key {gtsam.Symbol(key).string()} is not found in currentEstimate of priorValues")

        reg = fgtools.ScanMatch(pcSource, pcTarget, np.eye(4), showResult=self.params.loopClosure.showScanMatchResult, voxel_size=self.params.icp.voxelSize)

        fit = reg.fitness
        transFinal = reg.transformation
        if self.params.loopClosure.debug:
            print(f"Loop Check between keys {gtsam.Symbol(currKey).string()} and {gtsam.Symbol(key).string()} fitness: {fit:.4f} inliers: {reg.inlier_rmse:.4f}")

        if fit > self.params.loopClosure.fitnessThreshold:
            cs = gtsam.Symbol(currKey)
            ts = gtsam.Symbol(key)
            if self.params.loopClosure.debug:
                print(f"Loop Found between keys {chr(cs.chr()) if cs.chr() else ''}{cs.index()} and {chr(ts.chr()) if ts.chr() else ''}{ts.index()}")
            loopNoise = gtsam.noiseModel.Diagonal.Variances(np.asarray([max(1-fit, 1e-2) * self.params.loopClosure.noiseScale] * 6))

            Twrong = self.currentEstimate_.atPose3(currKey).matrix()

            Tfixed = transFinal @ Twrong
            posefixed = gtsam.Pose3(Tfixed)
            # draw_registration_result(self.pcds_[currKey], pcTarget, Tfixed, "Fixed Tranform")
            # if self.isInitialized_ and addFactor:
            if addFactor:
                self.AddFactor(gtsam.BetweenFactorPose3(currKey, key, posefixed.between(poseTarget), loopNoise))
                if self.IsCurrentSessionKey(currKey) and not self.IsCurrentSessionKey(key):
                    fgtools.AddValueToDictList(self.currentToPriorAnchors_, currKey, key)

        return reg
    
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

    def DecrementSize_legacy(self, finalSize:int|None=None, char:str|None=None, removeAll:bool=False) -> list[int]:
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
            self.timestamps_.pop(k)
        
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
            self.timestamps_.pop(k)
        # self.UpdateISAM2(reset=True)
        self.UpdateSmoother()
    
    def GetSessionInfo(self, excludeSymbols: list[str]|None = None) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values, dict[int, o3d.geometry.PointCloud], dict[int, float]]:
        if excludeSymbols is None:
            excludeSymbols = []

        # Remove all current session keys from the graph, values, pcds, and timestamps
        finalGraph = gtsam.NonlinearFactorGraph()
        tmpGraph = self.smoother_.getFactors()
        for i in range(tmpGraph.size()):
            f = tmpGraph.at(i)
            if not f:
                continue
            if not self.ContainsKeyWithSymbols(f.keys(), excludeSymbols):
                finalGraph.push_back(f)

        finalValues = gtsam.Values()
        finalPcds = {}
        finalTimestamps = {}
        for k in self.values.keys():
            s = gtsam.Symbol(k).chr()
            if chr(s) not in excludeSymbols:
                finalValues.insert(k, self.values.atPose3(k))
                finalPcds[k] = self.pcds_[k]
                finalTimestamps[k] = self.timestamps_[k]

        return finalGraph, finalValues, finalPcds, finalTimestamps
    
    def GetFinalSessionInfo(self) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values, dict[int, o3d.geometry.PointCloud], dict[int, float]]:
        '''Same as GetSessionInfo but excludes the fixedLagSmoother window if applicable.'''
        if self.params.fixedLagCurrentSession:
            return self.GetSessionInfo(excludeSymbols=[self.currentSessionSymbol_])
        else:
            return self.GetSessionInfo()

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
            self.RenameKeys(chars=self.params.priorSessionSymbolChrs,
                            autoRename=self.params.autoRename,
                            updateOptimization=True,
                            resetOptimization=True)
        else:
            self.UpdateSmoother(reset=True)

    def SaveSession(self, directory:str|Path|None=None):
        if directory is None:
            directory = self.params.saveSessionDir

        directory = Path(directory)
        while directory.exists():
            res = input(f"Directory {directory} already exists. Overwrite? (y/[n]): ")
            if res.lower()[0] == "y":
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
                        if confirm.lower()[0] == "y":
                            directory = Path(res)
        print(f"Saving session to {directory}...")
        os.makedirs(directory, exist_ok=True)

        finalGraph, finalValues, finalPcds, finalTimestamps = self.GetFinalSessionInfo()
        # fgtools.writeG2oWithPrior(self.graph, self.values, str(directory / "graph.g2o"))
        if not gtsam_unstable.serializeGraphToFile(finalGraph, str(directory / "graph.gtsam")):
            print(f"Error saving graph to {directory / 'graph.gtsam'}")
        if not gtsam_unstable.serializeValuesToFile(finalValues, str(directory / "values.gtsam")):
            print(f"Error saving values to {directory / 'values.gtsam'}")
        if not fgtools.savePointClouds(finalPcds, str(directory / "scans"), remove_existing=False):
            print(f"Error saving point clouds to {directory / 'scans'}")
        if not fgtools.saveOdomTUM(finalTimestamps, finalValues, str(directory / "odom_tum.txt")):
            print(f"Error saving odometry to {directory / 'odom_tum.txt'}")
        if not fgtools.saveTimestampsWithKeys(finalTimestamps, str(directory / "timestamps.txt")):
            print(f"Error saving timestamps to {directory / 'timestamps.txt'}")
        print(f"Done saving session.")

    def LoadSession(self, directory:str|Path|None=None):
        if directory is None:
            directory = self.params.loadSessionDir
            if directory is None:
                raise ValueError("LoadSession: No directory provided and params.loadSessionDir is None.")

        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        
        graphFile = directory / "graph.gtsam"
        valuesFile = directory / "values.gtsam"
        scansDir = directory / "scans"
        odomFile = directory / "odom_tum.txt"
        timestampsFile = directory / "timestamps.txt"

        if not graphFile.exists() or not valuesFile.exists() or not scansDir.exists() or not odomFile.exists() or not timestampsFile.exists():
            raise FileNotFoundError(f"One or more files in {directory} are missing. Required files: graph.gtsam, values.gtsam, scans/, odom_tum.txt, timestamps.txt")

        self.priorGraph_ = gtsam_unstable.deserializeGraphFromFile(str(graphFile))
        self.priorValues_ = gtsam_unstable.deserializeValuesFromFile(str(valuesFile))
        self.pcds_ = fgtools.loadPosePCDs(str(scansDir))
        self.timestamps_ = fgtools.loadTimestampsWithKeys(str(timestampsFile))


        valuesKeys = set(self.priorValues_.keys())
        if valuesKeys != set(self.pcds_.keys()) or valuesKeys != set(self.timestamps_.keys()):
            raise ValueError(f"LoadSession: Keys in values, scans, and timestamps do not match. \
                             # Value keys: {len(self.priorValues_.keys())}, # Scan keys: {len(self.pcds_.keys())}, \
                             # Timestamp keys: {len(self.timestamps_.keys())}")
        
        if self.params.rekeyPrior:
            self.priorGraph_, self.priorValues_ = self.RenameKeys(chars=self.params.priorSessionSymbolChrs,
                                                                  autoRename=self.params.autoRename,
                                                                  updateOptimization=False)
            
        self.priorIsam_ = gtsam.ISAM2()
        self.priorIsam_.update(self.priorGraph_, self.priorValues_)
        self.priorIsam_.update()

        self.UpdateSmoother(reset=True)
        self.UpdatePriorTree()
        self.UpdatePriorKeyToFactorIndexDict()
            
    def Update(self, pointCloud: o3d.geometry.PointCloud, pose:np.ndarray|gtsam.gtsam.Pose3|None=None, relPose:gtsam.Pose3|None=None, timestamp:float=0.0):
        '''
        Assumes pose is being added to last value in graph
        '''

        if isinstance(pose, np.ndarray):
            pose = pose.flatten()
            assert len(pose) == 7, "Pose must be in format [tx, ty, tz, qw, qx, qy, qz]"
            pose = gtsam.Pose3(gtsam.Rot3(pose[3], pose[4], pose[5], pose[6]), pose[:3])

        if not self.updateInitialized_:
            if pose is None:
                raise ValueError("Pose must be provided for the first keyframe")
            self.updateInitialized_ = True
            print("Adding Prior Factor to Current Session")
            key = gtsam.Symbol(self.params.currSessionSymbolChr,0).key()
            if self.priorGraph_.size() != 0:
                self.AddPrior(key, pose, gtsam.noiseModel.Diagonal.Sigmas(np.array([np.pi**2, np.pi**2, np.pi**2, 10, 10, 10])))
            else:
                self.AddPrior(key, pose, gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2]*6)))
            self.AddPointCloud(key, pointCloud, timestamp=timestamp)
        else:
            key = self.AddOdomFactor(pose, relPose)
            self.AddPointCloud(key, pointCloud, timestamp=timestamp)
        
        self.loopCnt_ += 1
        if self.loopCnt_ > self.params.loopClosure.skipFrames:
            self.loopCnt_ = 0
            # self.currentSession_.CheckLoopClosure(inter=False)

            self.CheckLoopClosures()

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
        # if self.priorSessionSymbols_ is None:
        #     return []
        return deepcopy(self.priorSessionSymbols_)
    
    @property
    def isInitialized(self) -> bool:
        return self.isInitialized_

    @property
    def lastPose(self) -> gtsam.gtsam.Pose3:
        if self.currentEstimate_.exists(self.lastOdomKey_):
            return self.currentEstimate_.atPose3(self.lastOdomKey_)
        else:
            raise ValueError(f"Last odom key {gtsam.Symbol(self.lastOdomKey_).string()} not found in currentEstimate or priorValues")