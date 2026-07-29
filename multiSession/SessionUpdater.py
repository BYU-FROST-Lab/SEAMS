#!/usr/bin/env python3

import numpy as np
np.set_printoptions(linewidth=np.inf, suppress=True)
from copy import copy, deepcopy
import time

import open3d as o3d

# Local imports
import gtsam
from multiSession.SessionManager import SessionManager
from params import Params


class SessionUpdater:
    def __init__(self, priorSession: SessionManager=SessionManager(), params:Params=Params()):
        self.params = params
        # self.AddPriorSession(priorSession)    
        # self.currentSession_ = SessionManager()

        self.combinedSession_ = SessionManager(params=params)
        if priorSession.Size() > 0:
            self.combinedSession_.SetPriorGraphAndValues(priorSession.fullGraph, priorSession.fullValues, self.params.priorSessionSymbolChrs)
            self.combinedSession_.SetPointClouds(priorSession.scans)
        self.lastPose_ = None
        self.initialized_ = False
        self.loopCnt_ = 0

        self.fixedLagKeep = self.params.fixedLagKeepFrames

        self.DEFAULT_NOISE6 = gtsam.noiseModel.Diagonal.Sigmas(np.asarray([1e-6] * 6))
        self.DEFAULT_NOISE3 = gtsam.noiseModel.Diagonal.Sigmas(np.asarray([1e-6] * 3))

    def AddPriorSession(self, priorSession: SessionManager):
        pass
        # if self.params.rekeyPrior:
        #     priorSession.RenameKeys(self.params.priorSessionSymbolChrs, resetIsam2=True)
        #     self.params.priorSessionSymbolChrs = priorSession.priorSessionSymbols
        #     self.params.addToPriorSymbolChr = priorSession.addToPriorSymbol
        # # self.priorSession_ = priorSession

    def AddInitialOrientationForCurrentSession(self, rot: gtsam.Rot3):
        if isinstance(rot, gtsam.Rot3):
            self.initialRot = rot
        else:
            raise TypeError(f"Input must by of type gtsam.Rot3 not {type(rot)}")

    def AddOdomKeyFrame(self, pointCloud: o3d.geometry.PointCloud, pose:np.ndarray|gtsam.gtsam.Pose3|None=None, relPose:gtsam.Pose3|None=None, timestamp:float=0.0):
        '''
        Assumes pose is being added to last value in graph
        '''

        if isinstance(pose, np.ndarray):
            pose = pose.flatten()
            assert len(pose) == 7, "Pose must be in format [tx, ty, tz, qw, qx, qy, qz]"
            pose = gtsam.Pose3(gtsam.Rot3(pose[3], pose[4], pose[5], pose[6]), pose[:3])

        if not self.initialized_:
            if pose is None:
                raise ValueError("Pose must be provided for the first keyframe")
            self.initialized_ = True
            print("Adding Prior Factor to Current Session")
            if hasattr(self, "initialRot"):
                print(f"Applying initial rotation to prior: {self.initialRot}")
                finalRot = self.initialRot * pose.rotation()
                pose = gtsam.Pose3(finalRot, pose.translation())
            key = gtsam.Symbol(self.params.currSessionSymbolChr,0).key()
            if self.combinedSession_.priorIsam_.size() != 0:
                self.combinedSession_.AddPrior(key, pose, gtsam.noiseModel.Diagonal.Variances(np.array([np.pi**2, np.pi**2, np.pi**2, 10, 10, 10])))
            else:
                self.combinedSession_.AddPrior(key, pose, gtsam.noiseModel.Diagonal.Variances(np.array([1e-2]*6)))
            self.combinedSession_.AddPointCloud(key, pointCloud, timestamp=timestamp)
            return
        
        key = self.combinedSession_.AddOdomFactor(pose, relPose)
        self.combinedSession_.AddPointCloud(key, pointCloud, timestamp=timestamp)
        
        self.loopCnt_ += 1
        if self.loopCnt_ > self.params.loopClosure.skipFrames:
            self.loopCnt_ = 0
            # self.currentSession_.CheckLoopClosure(inter=False)

            self.combinedSession_.CheckLoopClosures()
        

    def UpdateFixedLagKeep(self, num:int):
        self.fixedLagKeep = num

    # @property
    # def prior(self) -> SessionManager:
    #     return self.priorSession_
    
    # @property
    # def current(self) -> SessionManager:
    #     return self.currentSession_
    
    @property
    def combined(self) -> SessionManager:
        return self.combinedSession_


        
        

        
 
        

        


    

