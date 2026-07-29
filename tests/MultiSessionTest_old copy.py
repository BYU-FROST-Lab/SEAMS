#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
np.set_printoptions(linewidth=np.inf, suppress=False)
import copy
import time
from pathlib import Path
import open3d as o3d
from scipy.spatial import KDTree

# Local imports
import gtsam
import common.FGVizHelpers as fgtools
from multiSession.SessionUpdater import SessionUpdater
from multiSession.SessionManager import SessionManager
from params import Params
from common.customTypes import DecisionType, FixedLagMarginalizationType
from common.LiveMapVisualizer import LiveMapVisualizer
from common.kissicp_odom import KissICPOdom
import common.FGVizHelpers as fgtools
from params.params import Params

# ------------------------------------------------------------
# Visualization flags
# ------------------------------------------------------------
ENABLE_MATPLOTLIB = True
ENABLE_OPEN3D = False



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
session1 = SessionManager(directory1 / fileName, loadGPS=True)
session1.UpdateISAM2()
# session1 = SessionManager()

session2 = SessionManager(directory2 / fileName, loadGPS=True)
session2.UpdateISAM2()
# session3 = SessionManager(directory3 / fileName, loadGPS=True)

# session3.UpdateISAM2()

session1.LoadScans(directory1 / "Scans")
# Use blocks for session 1
# graph, values = fgtools.readG2oWithPrior(str(directory1 / "sparseGraph.g2o"), loadGPS=False)
# graph.add(gtsam.PriorFactorPose3(0, values.atPose3(0), gtsam.noiseModel.Diagonal.Variances(np.array([1e-8]*6))))
# session1.SetPriorGraphAndValues(graph, values, ["A"])
# block_pcds = fgtools.loadBlockPCDs(str(directory1 / "blocks/"))
# non_block_pcds = fgtools.loadPosePCDs(str(directory1 / "non_blocks/"))
# block2pose_mapping = fgtools.loadPose2BlockMapping(str(directory1 / "pose2block_mapping.txt"), reverse=True)
# pointClouds = {}
# for block_key, pcd in block_pcds.items():
#     pointClouds[gtsam.Symbol("A", block2pose_mapping[block_key]).key()] = pcd
# session1.SetPointClouds(pointClouds)

# combinedPointCloud = session1.CreateGlobalPCD()
# o3d.visualization.draw_geometries([combinedPointCloud])

session2.LoadScans(directory2 / "Scans")
# session3.LoadScans(directory3 / "Scans")

bag_colors = ['tab:blue', 'tab:orange', 'tab:green']

# ax = fgtools.plot_graph(session1.graph, session1.values, color_override=bag_colors[0], alpha=0.5, addKeyLabels=True)
# # fgtools.plot_graph_covariances(session1.values.keys(), session1.graph, session1.values, ax=ax)

# # fgtools.plot_graph(session2.graph, session2.orig_values_, ax=ax, color_override=bag_colors[1], alpha=0.5)
# # fgtools.plot_graph_covariances(session2.values.keys(), session2.graph, session2.values, ax=ax)

# # fgtools.plot_graph(session3.graph, session3.values, ax=ax, color_override=bag_colors[2], alpha=0.5)
# # fgtools.plot_graph_covariances(session3.values.keys(), session3.graph, session3.values, ax=ax)

# legendElements = [
#                     Line2D([0],[0], marker='o', color='w', label="bag 1", markerfacecolor=bag_colors[0]),
#                     # Line2D([0],[0], marker='o', color='w', label="bag 2", markerfacecolor=bag_colors[1]),
#                     # Line2D([0],[0], marker='o', color='w', label="bag 3", markerfacecolor=bag_colors[2]),
#                 ]
# ax.legend(handles=legendElements, loc='lower right')
# ax.set_title("Aligned Via GPS Only")

# plt.show()

# globalMap1 = session1.CreateGlobalPCD((1, 0, 0))
# globalMap2 = session2.CreateGlobalPCD((0, 1, 0))
# globalMap3 = session3.CreateGlobalPCD((0, 0, 1))

# o3d.visualization.draw_geometries([globalMap1, globalMap2, globalMap3])

#########################################################################################
# Session 2
#########################################################################################

priorSession = SessionManager()
priorSession = session1
# priorPoses, keys = GetPosesFromGraphEstimate(priorSession.values)
# priorSession.AddPrior(0, 
#                       gtsam.Pose3(
#                           gtsam.Rot3(priorPoses[0,3], priorPoses[0,4], priorPoses[0,5], priorPoses[0,6]), 
#                           priorPoses[0,:3]), 
#                       gtsam.noiseModel.Diagonal.Variances(np.array([1e8]*6)),
#                       skipValue=True
#                       )
updater = SessionUpdater(priorSession, 
                         Params(
                             rekeyPrior=True, 
                             currSessionSymbolChr='c',
                             addToPriorSymbolChr='t',
                             priorSessionSymbolChrs=['A'],
                             autoRename=True,
                             policyType=DecisionType.FITNESS,
                             fixedLagCurrentSession=True,
                             fixedLagMarg = FixedLagMarginalizationType.DENSE,
                            #  fixedLagMarg = FixedLagMarginalizationType.APPROX_PRIOR,
                             ),
                        )

_, ax = plt.subplots()

currentSession = session2
poses, keys = fgtools.GetPosesFromGraphEstimate(currentSession.fullValues)
scans = currentSession.scans
cnt = 0
totalCnt = 0
MAX_COUNT = 5
FIXED_WINDOW = False
startTime = time.time()

sessionSymbols = {ord('A'): 'C0', ord('B'): 'C2', ord('c'): 'C1'}

odom = KissICPOdom(directory2 / "Scans")

if ENABLE_OPEN3D:
    viz2 = LiveMapVisualizer(updater, window_name="Session 2 — Live Map")
    viz2.start()

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
    if i == 0:
        print(currPose)
    print(f"Total Count: {totalCnt}")
    if totalCnt == 268:
        pass
    updater.AddOdomKeyFrame(scans[keys[i]], currPose, relPose)

    # odom.update_last_pose(updater.combined.values.atPose3(updater.combined.values.keys()[-1]))
    if cnt >= MAX_COUNT: # or totalCnt > 225:
    # if True:
        cnt = 0
        if ENABLE_MATPLOTLIB:
            ax.clear()
            # fgtools.plot_graph(updater.prior.graph, updater.prior.values, ax=ax, color_override=bag_colors[0], alpha=0.5)
            # fgtools.plot_graph(updater.current.graph, updater.current.values, ax=ax, title="Testing Loop Closure in Graph", color_override=bag_colors[1], alpha=0.5)
            tmpValues = updater.combined.values
            # fgtools.plot_graph(updater.combined.priorGraph, updater.combined.priorValues, color_override="tab:grey", ax=ax, title="Testing Combined Graph", changeColorBySymbols=False)
            # fgtools.plot_graph_covariances(updater.combined.priorValues.keys(), updater.combined.priorGraph, updater.combined.priorValues, ax=ax, edgecolor="tab:grey")
            fgtools.plot_graph(updater.combined.graph, tmpValues, ax=ax, title="Testing Combined Graph", changeColorBySymbols=True, addKeyLabels=False, symbolColors=sessionSymbols)
            fgtools.plot_graph_covariances(tmpValues.keys(), updater.combined.fixedLagSmoother, tmpValues, ax=ax, changeColorBySymbols=True, symbolColors=sessionSymbols)
            
            if FIXED_WINDOW:
                p1 = tmpValues.atPose3(tmpValues.keys()[-1])
                p2 = tmpValues.atPose3(tmpValues.keys()[-min(Params().fixedLagKeepFrames+1, totalCnt+1)])
                offset = 5
                ax.set_xlim(min(p1.x(), p2.x())-offset, max(p1.x(), p2.x())+offset)
                ax.set_ylim(min(p1.y(), p2.y())-offset, max(p1.y(), p2.y())+offset)
            plt.pause(0.05)
        if ENABLE_OPEN3D and not viz2.update():
            print("Open3D Failed: Session 2 window closed. Matplotlib will be used instead.")
            viz2.close()
            ENABLE_OPEN3D = False
            ENABLE_MATPLOTLIB = True
    cnt += 1
    totalCnt += 1

if ENABLE_OPEN3D:
    viz2.close()

print(f"Total time for second session: {time.time() - startTime}")

ax.clear()
fgtools.plot_graph(updater.combined.priorGraph, updater.combined.priorValues, color_override="tab:grey", ax=ax, title="Testing Combined Graph")
fgtools.plot_graph_covariances(updater.combined.priorValues.keys(), updater.combined.priorGraph, updater.combined.priorValues, ax=ax, edgecolor="tab:grey")
fgtools.plot_graph(updater.combined.graph, updater.combined.values, ax=ax, title="Testing Combined Graph", addKeyLabels=False, changeColorBySymbols=True, symbolColors=sessionSymbols)
fgtools.plot_graph_covariances(updater.combined.values.keys(), updater.combined.fixedLagSmoother, updater.combined.values, ax=ax, changeColorBySymbols=True, symbolColors=sessionSymbols)

plt.show()
print("Done with second session")
# updater.combined.Finalize()
# combinedPointCloud = updater.combined.CreateGlobalPCD()
# o3d.visualization.draw_geometries([combinedPointCloud])




# ###################################################################################
# # Session 3
# ###################################################################################

# priorSession = updater.combined
# # priorSession = session1
# priorPoses, keys = fgtools.GetPosesFromGraphEstimate(priorSession.values)
# # priorSession.AddPrior(0, 
# #                       gtsam.Pose3(
# #                           gtsam.Rot3(priorPoses[0,3], priorPoses[0,4], priorPoses[0,5], priorPoses[0,6]), 
# #                           priorPoses[0,:3]), 
# #                       gtsam.noiseModel.Diagonal.Variances(np.array([1e-8]*6)),
# #                       skipValue=True
# #                       )
# updater = SessionUpdater(priorSession, 
#                          Params(
#                              rekeyPrior=True, 
#                              policyType=DecisionType.DISTANCE,
#                              fixedLagCurrentSession=True,
#                              )
#                          )
# # updater.AddInitialOrientationForCurrentSession(gtsam.Rot3(w=0., x=0., y=0., z=1.))

# _, ax = plt.subplots()

# currentSession = session3
# poses, keys = fgtools.GetPosesFromGraphEstimate(currentSession.values)
# scans = currentSession.scans
# cnt = 0

# if ENABLE_OPEN3D:
#     viz3 = LiveMapVisualizer(updater, window_name="Session 3 — Live Map")
#     viz3.start()

# for i, pose in enumerate(poses):
#     currPose = gtsam.Pose3(gtsam.Rot3(pose[3], pose[4], pose[5], pose[6]), pose[:3])
#     if i == 0:
#         relPose = None
#     else:
#         tmpPose = poses[i-1]
#         prevPose = gtsam.Pose3(gtsam.Rot3(tmpPose[3], tmpPose[4], tmpPose[5], tmpPose[6]), tmpPose[:3])
#         relPose = prevPose.between(currPose)
#     updater.AddOdomKeyFrame(pose, scans[keys[i]], relPose)

#     if cnt > 5:
#         cnt = 0
#         if ENABLE_MATPLOTLIB:
#             ax.clear()
#             # fgtools.plot_graph(updater.prior.graph, updater.prior.values, ax=ax, color_override=bag_colors[0], alpha=0.5)
#             # fgtools.plot_graph(updater.current.graph, updater.current.values, ax=ax, title="Testing Loop Closure in Graph", color_override=bag_colors[1], alpha=0.5)
#             fgtools.plot_graph(updater.combined.graph, updater.combined.values, ax=ax, title="Testing Combined Graph", changeColorBySymbols=True)
#             plt.pause(0.05)
#         if ENABLE_OPEN3D and not viz3.update():
#             print("Open3D Failed: Session 3 window closed. Matplotlib will be used instead.")
#             viz3.close()
#             ENABLE_OPEN3D = False
#             ENABLE_MATPLOTLIB = True
#     cnt += 1

# if ENABLE_OPEN3D:
#     viz3.close()

# ax.clear()
# # fgtools.plot_graph(updater.prior.graph, updater.prior.values, ax=ax, color_override=bag_colors[0], alpha=0.5)
# # fgtools.plot_graph(updater.current.graph, updater.current.values, ax=ax, title="Testing Loop Closure in Graph", color_override=bag_colors[1], alpha=0.5, addKeyLabels=True)
# fgtools.plot_graph(updater.combined.graph, updater.combined.values, ax=ax, title="Testing Combined Graph", addKeyLabels=False, changeColorBySymbols=True)
# # fgtools.plot_graph_covariances(updater.combined.values.keys(), updater.combined.graph, updater.combined.values, ax=ax)
# plt.show()

# combinedPointCloud = updater.combined.CreateGlobalPCD()
# o3d.visualization.draw_geometries([combinedPointCloud])


