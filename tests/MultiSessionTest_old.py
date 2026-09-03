#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(linewidth=np.inf, suppress=False)
import time
from pathlib import Path
import open3d as o3d

# Local imports
import gtsam
import common.FGVizHelpers as fgtools
from multiSession.SessionManager import SessionManager
from params import Params
from common.customTypes import DecisionType, FixedLagMarginalizationType
from common.kissicp_odom import KissICPOdom
import common.FGVizHelpers as fgtools
from params.params import Params

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
scriptDir = Path(__file__).resolve().parents[2]
directory1 = scriptDir  / "results/urban/around_block_initial/"
# directory1 = scriptDir  / "python/TestResults/"
directory2 = scriptDir / "results/urban/around_block_short_reconnect/"
directory3 = scriptDir / "results/urban/quick_map_disconnect/"

fileName = "fullSavedGraph.g2o"
useGPS = True
session1 = SessionManager()
session1.LoadPriorGraphAndValues(str(directory1 / fileName), loadGPS=True)
# session1.UpdateISAM2()
# session1 = SessionManager()

session2 = SessionManager()
session2.LoadPriorGraphAndValues(str(directory2 / fileName), loadGPS=True)
# session2.UpdateISAM2()


session1.LoadScans(directory1 / "Scans")


session2.LoadScans(directory2 / "Scans")

bag_colors = ['tab:blue', 'tab:orange', 'tab:green']



#########################################################################################
# Session 2
#########################################################################################

# priorSession = SessionManager()
# priorSession = session1
# priorPoses, keys = GetPosesFromGraphEstimate(priorSession.values)
# priorSession.AddPrior(0, 
#                       gtsam.Pose3(
#                           gtsam.Rot3(priorPoses[0,3], priorPoses[0,4], priorPoses[0,5], priorPoses[0,6]), 
#                           priorPoses[0,:3]), 
#                       gtsam.noiseModel.Diagonal.Variances(np.array([1e8]*6)),
#                       skipValue=True
#                       )

params = Params(rekeyPrior=True, 
                currSessionSymbolChr='c',
                addToPriorSymbolChr='t',
                priorSessionSymbolChrs=['A'],
                autoRename=True,
                policyType=DecisionType.INLIER_RATIO,
                # fixedLagMarg = FixedLagMarginalizationType.DENSE,
                fixedLagMarg = FixedLagMarginalizationType.APPROX_PRIOR,
                )

graph = session1.graph
values = session1.values
new_graph = gtsam.NonlinearFactorGraph()
for i in range(graph.size()):
    f = graph.at(i)
    if isinstance(f, gtsam.GPSFactor):
        continue
    if isinstance(f, gtsam.PriorFactorPose3):
        key = f.keys()[0]
        new_pose = values.atPose3(key)
        noise = f.noiseModel()
        new_graph.add(gtsam.PriorFactorPose3(key, new_pose, noise))
    else:
        new_graph.add(f)


currentSession = SessionManager(params=params)
currentSession.SetPriorGraphAndValues(new_graph, values)
currentSession.SetPointClouds(session1.scans)
currentSession.SetTimestamps(session1.timestamps) # May need to fix this. Not sure if session1 has timestamps

_, ax = plt.subplots()

poses, keys = fgtools.GetPosesFromGraphEstimate(session2.fullValues)
scans = session2.scans
cnt = 0
totalCnt = 0
MAX_COUNT = 5
FIXED_WINDOW = False
startTime = time.time()
relPoses = []

sessionSymbols = {ord('A'): 'C0', ord('B'): 'C2', ord('c'): 'C1'}

odom = KissICPOdom(directory2 / "Scans")


for i, pose in enumerate(poses):
    if i == 0:
        currPose = gtsam.Pose3(gtsam.Rot3(pose[3], pose[4], pose[5], pose[6]), pose[:3])
        print(currPose)
        odom.set_initial_pose(currPose)
    #     relPose = None
    # else:
    #     tmpPose = poses[i-1]
    #     prevPose = gtsam.Pose3(gtsam.Rot3(tmpPose[3], tmpPose[4], tmpPose[5], tmpPose[6]), tmpPose[:3])
    #     relPose = prevPose.between(currPose)
    currPose, relPose, deskewed_points = odom.process_next_keyframe()
    # if relPose:
    #     relPoses.append(relPose)
    #     print(f"Frame {totalCnt}: relative pose computed")
    #     print(f"    Translation: {relPose.translation()}, Rotation: {relPose.rotation().rpy()}")
    if i == 0:
        print(currPose)
    print(f"Total Count: {totalCnt}")
    if totalCnt == 51:
        pass
    if relPose is None:
        currentSession.Update(scans[keys[i]], currPose, None, float(i))
    else:
        currentSession.Update(scans[keys[i]], None, relPose, float(i))

    # odom.update_last_pose(currentSession.values.atPose3(currentSession.values.keys()[-1]))
    if cnt >= MAX_COUNT: # or totalCnt > 225:
    # if True:
        cnt = 0
        ax.clear()
        # fgtools.plot_graph(updater.prior.graph, updater.prior.values, ax=ax, color_override=bag_colors[0], alpha=0.5)
        # fgtools.plot_graph(updater.current.graph, updater.current.values, ax=ax, title="Testing Loop Closure in Graph", color_override=bag_colors[1], alpha=0.5)
        tmpValues = currentSession.values
        # fgtools.plot_graph(currentSession.priorGraph, currentSession.priorValues, color_override="tab:grey", ax=ax, title="Testing Combined Graph", changeColorBySymbols=False)
        # fgtools.plot_graph_covariances(currentSession.priorValues.keys(), currentSession.priorGraph, currentSession.priorValues, ax=ax, edgecolor="tab:grey")
        fgtools.plot_graph(currentSession.graph, tmpValues, ax=ax, title="Testing Combined Graph", changeColorBySymbols=True, addKeyLabels=False, symbolColors=sessionSymbols)
        fgtools.plot_graph_covariances(tmpValues.keys(), currentSession.fixedLagSmoother, tmpValues, ax=ax, changeColorBySymbols=True, symbolColors=sessionSymbols)
        
        if FIXED_WINDOW:
            p1 = tmpValues.atPose3(tmpValues.keys()[-1])
            p2 = tmpValues.atPose3(tmpValues.keys()[-min(Params().fixedLagKeepFrames+1, totalCnt+1)])
            offset = 5
            ax.set_xlim(min(p1.x(), p2.x())-offset, max(p1.x(), p2.x())+offset)
            ax.set_ylim(min(p1.y(), p2.y())-offset, max(p1.y(), p2.y())+offset)
        plt.pause(0.05)

    cnt += 1
    totalCnt += 1



print(f"Total time for second session: {time.time() - startTime}")

ax.clear()
fgtools.plot_graph(currentSession.priorGraph, currentSession.priorValues, color_override="tab:grey", ax=ax, title="Testing Combined Graph")
fgtools.plot_graph_covariances(currentSession.priorValues.keys(), currentSession.priorGraph, currentSession.priorValues, ax=ax, edgecolor="tab:grey")
fgtools.plot_graph(currentSession.graph, currentSession.values, ax=ax, title="Testing Combined Graph", addKeyLabels=False, changeColorBySymbols=True, symbolColors=sessionSymbols)
fgtools.plot_graph_covariances(currentSession.values.keys(), currentSession.fixedLagSmoother, currentSession.values, ax=ax, changeColorBySymbols=True, symbolColors=sessionSymbols)

plt.show()
print("Done with second session")
# currentSession.Finalize()
# combinedPointCloud = currentSession.CreateGlobalPCD()
# o3d.visualization.draw_geometries([combinedPointCloud])

