#!/usr/bin/env python3

from operator import gt

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
np.set_printoptions(linewidth=np.inf, suppress=False)
import copy
import time
from pathlib import Path
import open3d as o3d
from scipy.spatial import KDTree
import pickle


# Local imports
import gtsam
import gtsam_unstable
import common.FGVizHelpers as fgtools
from multiSession.SessionManager import SessionManager
from params import Params
from common.customTypes import DecisionType, FixedLagMarginalizationType
from common.LiveMapVisualizer import LiveMapVisualizer
from common.kissicp_odom import KissICPOdom
from params.params import Params
from ros2.readFromRosbags import ReadTopicsFromRosbag, GroundTruthLoader
from sparcification.NFR import GraphSparsification

# ------------------------------------------------------------
# Visualization flags — override these or use --viz on the CLI
# ------------------------------------------------------------
ENABLE_MATPLOTLIB = True
ENABLE_OPEN3D = False


moduleDir = Path(__file__).resolve().parents[1]

def odom_to_robot_frame(last_odom_pose: gtsam.Pose3, last_robot_pose: gtsam.Pose3) -> gtsam.Pose3:
    '''
    Computes the tranformation matrix from the odometry frame to the robot's frame given the last known robot pose.
    
    :params last_odom_pose: Pose in the odometry frame
    :params last_robot_pose: Last known robot pose
    '''

    # Compute the transformation from odometry frame to robot frame
    T_odom_to_robot = last_robot_pose.between(last_odom_pose)
    return T_odom_to_robot


def run_session(params: Params = Params(), enable_matplotlib: bool = ENABLE_MATPLOTLIB, enable_open3d: bool = ENABLE_OPEN3D):
    #########################################################################################
    # Session Parameters
    #########################################################################################

    MAX_COUNT = 5
    FIXED_WINDOW = False
    SAVE_FULL_SESSION = False
    SAME_SESSION_COV_CHANGE_PLOT = False
    SAVE_COVARIANCE_OF_CURRENT_POSE = False
    SAVE_UPDATE_TIMING = False

    show_mpl = enable_matplotlib
    show_o3d = enable_open3d

    #########################################################################################
    priorSessionPath = params.loadSessionDir
    currentSession = SessionManager(priorSessionPath, params)

    bag1 = ReadTopicsFromRosbag(str(params.bagFile), waiting_threshold=0.0)

    if params.gtFile is not None:
        gt = GroundTruthLoader(str(params.gtFile))
    else:
        gt = None
    gt2 = GroundTruthLoader("/home/benb/data_byu/bags/urban/groundTruth/block0_gt/optimized_poses.txt")
        
    kissOdom = KissICPOdom(config_file_path=params.kissICPConfigFile)

    firstPass = True
    cnt = 0
    totalCnt = 0

    poses = []

    sessionSymbols = {ord('c'): 'C0'}#,ord('A'): 'C1', ord('B'): 'C2', ord('C'): 'C3'}

    if show_mpl:
        _, ax = plt.subplots()

    viz = None
    if show_o3d:
        viz = LiveMapVisualizer(currentSession, window_name="MultiSession — Live Map")
        viz.start()

    start_time = time.time()
    firstTimestamp = None
    foundFirstIntraLoop = False
    prevISAM2 = gtsam.ISAM2()
    sscp_dets: dict [int,list[float]] = {}
    whole_graph_info_dets: list[float] = []
    cov_current_pose_arr = []
    updateTimes = []

    for topic, timestamp, xyz_points in bag1.next_message():
        if firstTimestamp is None:
            firstTimestamp = timestamp

        
        if firstPass and gt is not None:
            gtPose = gt.get_closest_gt(timestamp)
            print(f"Initial pose is: \n{gtPose}")
            kissOdom.set_initial_pose(gtPose)
        composedPose_odom, relPose_odom, deskewed_points = kissOdom.process_next_keyframe(xyz_points) # Poses are returned in the odometry frame, not the robot's frame

        if firstPass:
            assert relPose_odom is None, "Relative pose should be None for the first keyframe"
            update_time = time.time()
            if gt is not None:
                # gtPose = gt.get_closest_gt(timestamp)
                # print(f"Initial pose is: \n{gtPose}")
                sessionResult = currentSession.Update(deskewed_points, gtPose, timestamp=timestamp)
                # poses.append(gtPose)
            else:
                sessionResult = currentSession.Update(deskewed_points, composedPose_odom, timestamp=timestamp)
                # poses.append(composedPose_odom)
            update_time = time.time() - update_time
            firstPass = False
            lastPose_odom = composedPose_odom
        else:
            # Filter out frames that are too close to the last pose
            relPose_tmp = lastPose_odom.between(composedPose_odom)
            if np.linalg.norm(relPose_tmp.translation()) < 1.0:# and relPose_tmp.rotation().rpy().max() < np.deg2rad(5):
                continue
            # T_odom_to_robot = odom_to_robot_frame(lastPose_odom, currentSession.values.atPose3(currentSession.values.keys()[-1]))
            # relPose_tmp = T_odom_to_robot.compose(relPose_tmp)
            # poses.append(relPose_tmp)
            update_time = time.time()
            sessionResult = currentSession.Update(deskewed_points, relPose=relPose_tmp, timestamp=timestamp)
            update_time = time.time() - update_time
            lastPose_odom = composedPose_odom
            # print(f"Frame {totalCnt}: relative pose computed")
            # print(f"    Translation: {relPose_tmp.translation()}, Rotation: {relPose_tmp.rotation().rpy()}")

        if totalCnt == 35:
            pass

        if SAME_SESSION_COV_CHANGE_PLOT:
            start = time.time()

            ################################
            # Get Covariance of first loop #
            ################################

            # if sessionResult.numIntraLoops > 0 and not foundFirstIntraLoop:
            #     foundFirstIntraLoop = True
            #     refKeys = currentSession.values.keys()[:-1]
            #     for k in refKeys:
            #         sscp_dets[k] = [np.linalg.det(prevISAM2.marginalInformation(k))] # Take determinant of translation part of covariance matrix

            # if foundFirstIntraLoop:
            #     currIsam = currentSession.isam2
            #     for k in refKeys:
            #         if k in sscp_dets:
            #             sscp_dets[k].append(np.linalg.det(currIsam.marginalCovariance(k))) # Take determinant of translation part of covariance matrix
            #         else:
            #             sscp_dets[k] = [np.linalg.det(currIsam.marginalCovariance(k))] # Take determinant of translation part of covariance matrix
            #     print(f"Time to calc full det: {time.time() - start}")

            ###############################
            # Get Covariance of New nodes #
            ###############################

            refKeys = currentSession.GetKeys(currentSession.addToPriorSymbol)
            currIsam = currentSession.isam2
            for k in refKeys:
                if k in sscp_dets:
                    sscp_dets[k].append(np.linalg.det(currIsam.marginalCovariance(k))) # Take determinant of translation part of covariance matrix
                else:
                    sscp_dets[k] = [np.linalg.det(currIsam.marginalCovariance(k))] # Take determinant of translation part of covariance matrix
            print(f"Time to calc full det: {time.time() - start}")

            start = time.time()
            currentSession.isam2.marginalInformation(k)
            print(f"Time to calc marg information from isam: {time.time() - start}")

            start = time.time()
            currentSession.smoother_.marginalCovariance(k)
            print(f"Time to calc marg information from isam: {time.time() - start}")
            


            ##################################
            # Get Information of Full window #
            ##################################

            # refKeys = currentSession.GetMostRecentKeys(params.fixedLagKeepFrames)
            # if refKeys:
            #     currIsam = currentSession.isam2
            #     if sscp_dets:
            #         sscp_dets[0].append(np.linalg.det(currIsam.jointMarginalInformation(refKeys).fullMatrix())) # Take determinant of translation part of covariance matrix
            #     else:
            #         sscp_dets[0] = [np.linalg.det(currIsam.jointMarginalInformation(refKeys).fullMatrix())] # Take determinant of translation part of covariance matrix

            # graph = currentSession.graph
            # values = currentSession.values
            # gfg = graph.linearize(values)
            # gbt = gfg.eliminateMultifrontal()
            # whole_graph_info_dets.append(gbt.logDeterminant())

            # print(f"Time to calc full det: {time.time() - start}")

        if SAVE_COVARIANCE_OF_CURRENT_POSE:
            cov = currentSession.smoother_.marginalCovariance(currentSession.lastOdomKey_)
            cov_current_pose_arr.append(cov)

        if SAVE_UPDATE_TIMING:
            updateTimes.append(update_time)

        # odom.update_last_pose(currentSession.values.atPose3(currentSession.values.keys()[-1]))
        if cnt >= MAX_COUNT: # or totalCnt > 225:
        # if True:
            cnt = 0

            if show_mpl:
                ax.clear()
                # fgtools.plot_graph(currentSession.prior.graph, currentSession.prior.values, ax=ax, color_override=bag_colors[0], alpha=0.5)
                # fgtools.plot_graph(currentSession.current.graph, currentSession.current.values, ax=ax, title="Testing Loop Closure in Graph", color_override=bag_colors[1], alpha=0.5)
                tmpValues = currentSession.values
                # fgtools.plot_graph(currentSession.priorGraph, currentSession.priorValues, color_override="tab:grey", ax=ax, title="Testing Combined Graph", changeColorBySymbols=False)
                # fgtools.plot_graph_covariances(currentSession.priorValues.keys(), currentSession.priorGraph, currentSession.priorValues, ax=ax, edgecolor="tab:grey")
                if gt is not None:
                    ax.plot(gt.array[:,1], gt.array[:,2], color='tab:green', label='Ground Truth', alpha=0.5)
                    # ax.plot(gt2.array[:,1], gt2.array[:,2], color='tab:orange', label='Ground Truth Block0', alpha=0.5)
                fgtools.plot_graph(currentSession.graph, tmpValues, ax=ax, title="Testing Combined Graph", changeColorBySymbols=True, addKeyLabels=False, symbolColors=sessionSymbols)
                # fgtools.plot_graph_covariances(tmpValues.keys(), currentSession.fixedLagSmoother, tmpValues, ax=ax, changeColorBySymbols=True, symbolColors=sessionSymbols)
                # odom_values = gtsam.Values()
                # for i, pose in enumerate(kissOdom.poses):
                #     odom_values.insert(gtsam.Symbol('o', i).key(), pose)
                # fgtools.plot_points(odom_values.keys(), odom_values, color='tab:red', ax=ax)
                
                if FIXED_WINDOW:
                    windowKeys = currentSession.GetMostRecentKeys(params.fixedLagKeepFrames+1)
                    p1 = tmpValues.atPose3(windowKeys[-1])
                    p2 = tmpValues.atPose3(windowKeys[0])
                    offset = 5
                    ax.set_xlim(min(p1.x(), p2.x())-offset, max(p1.x(), p2.x())+offset)
                    ax.set_ylim(min(p1.y(), p2.y())-offset, max(p1.y(), p2.y())+offset)
                plt.pause(0.05)

            if show_o3d and not viz.update():
                print("Open3D window closed — disabling live 3D visualization.")
                viz.close()
                viz = None
                show_o3d = False

        cnt += 1
        totalCnt += 1
        prevISAM2 = currentSession.isam2

    end_time = time.time()
    bag_time = timestamp - firstTimestamp
    run_time = end_time - start_time

    isRealTime = run_time < bag_time
    print(f"Real-time performance: {"\033[92m" if isRealTime else "\033[91m"}{isRealTime}\033[0m" \
            f" (bag time: {bag_time:.2f}s, run time: {run_time:.2f}s)")

    if show_o3d and viz is not None:
        viz.close()

    if show_mpl:
        ax.clear()
    else:
        ax = fgtools.plot_graph(gtsam.NonlinearFactorGraph(), gtsam.Values())
    
    final_graph, final_values, _, _ = currentSession.GetFinalSessionInfo()
    # if gt is not None:
    #     ax.plot(gt.array[:,1], gt.array[:,2], color='tab:green', label='Ground Truth', alpha=0.5)
    ax = fgtools.plot_graph(final_graph, final_values, ax=ax, title="Testing Combined Graph", addKeyLabels=False, changeColorBySymbols=True, symbolColors=sessionSymbols)
    # fgtools.plot_graph_covariances(final_values.keys(), final_graph, final_values, ax=ax, changeColorBySymbols=True, symbolColors=sessionSymbols)

    if SAVE_FULL_SESSION:
        currentSession.SaveSession(params.saveSessionDir)
        # Save matplotlib figure
        plt.savefig(params.saveSessionDir / "final_graph.png", dpi=300, bbox_inches='tight')
    else:
        Path.mkdir(params.saveSessionDir, exist_ok=True)

    if SAME_SESSION_COV_CHANGE_PLOT:
        # Save dictionary for plotting elsewhere
        with open(params.saveSessionDir / "det_of_addToPriorNodes.pkl", "wb") as f:
            pickle.dump(sscp_dets, f)   

    if SAVE_COVARIANCE_OF_CURRENT_POSE:
        with open(params.saveSessionDir / "cov_of_currentPoses.pkl", "wb") as f:
            pickle.dump(cov_current_pose_arr, f)

    if SAVE_UPDATE_TIMING:
        with open(params.saveSessionDir / "update_timing.pkl", "wb") as f:
                    pickle.dump(updateTimes, f)

    print("Done with session")
    
    # if show_mpl:
    plt.show()



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Run Multi-Session Test")
    # Argument for which dataset to use (current option is to use the Newer College dataset)
    parser.add_argument("-d", "--dataset", type=str, default="nc", choices=["nc", "urban"],help="Dataset to use (default: nc). Options: nc, urban")
    # Argument for which mission to run in the dataset
    parser.add_argument("-m", "--mission", type=str, default="quad-easy", help="Mission to run in the dataset.")
    # Argument for the load session directory (optional)
    parser.add_argument("-l", "--load_session_dir", type=str, default=None, help="Directory to load a previous session from (default: None)")
    # Argument for the save session directory (optional)
    parser.add_argument("-s", "--save_session_dir", type=str, default=None, help="Directory to save the current session to (default: None)")
    # Visualization mode
    parser.add_argument(
        "-v", "--viz",
        type=str,
        default=None,
        choices=["none", "mpl", "o3d", "both"],
        help="Visualization mode: none | mpl (matplotlib only) | o3d (Open3D only) | both. "
             "Defaults to the module-level ENABLE_MATPLOTLIB / ENABLE_OPEN3D flags.",
    )
    
    args = parser.parse_args()

    ########################################
    # Hardcode values for testing purposes #
    ########################################

    args.dataset = "nc"
    args.mission = "quad-medium"
    args.load_session_dir = moduleDir / "results/nc/quad-easy/"
    args.save_session_dir = moduleDir / "results/nc/quad-medium_cov_current_pose_eval"

    # args.dataset = "urban"
    # args.mission = "block1"
    # # args.load_session_dir = None
    # args.load_session_dir = moduleDir / "results/urban/block0/"
    # # args.save_session_dir = moduleDir / "results/urban/block0/"
    # args.save_session_dir = moduleDir / "results/urban/block1/"

    ########################################


    if args.dataset == "nc":
        nc_data_dir = Path("/home/benb/data_sets/NewerCollege/ros2")
        if args.mission == "quad-easy":
            bag_file = nc_data_dir / "quad-easy"
            gt_file = nc_data_dir / "ground_truth_tum" / "gt-nc-quad-easy.csv"
        elif args.mission == "quad-medium":
            bag_file = nc_data_dir / "quad-medium"
            gt_file = nc_data_dir / "ground_truth_tum" / "gt-nc-quad-medium.csv"
        elif args.mission == "quad-hard":
            bag_file = nc_data_dir / "quad-hard"
            gt_file = nc_data_dir / "ground_truth_tum" / "gt-nc-quad-hard.csv"
        else:
            raise ValueError(f"Unknown mission for nc dataset: {args.mission}. Options are: quad-easy, quad-medium, quad-hard")
    if args.dataset == "urban":
        urban_data_dir = Path("/home/benb/data_byu/bags/urban")
        if args.mission == "block0":
            bag_file = urban_data_dir / "block0"
            gt_file = urban_data_dir / "groundTruth" / "block0_gt" / "optimized_poses.txt"
        elif args.mission == "block1":
            bag_file = urban_data_dir / "block1"
            gt_file = urban_data_dir / "groundTruth" / "block1_gt" / "optimized_poses.txt"
        elif args.mission == "block2":
            bag_file = urban_data_dir / "block2"
            gt_file = urban_data_dir / "groundTruth" / "block2_gt" / "optimized_poses.txt"
        else:
            raise ValueError(f"Unknown mission for urban dataset: {args.mission}. Options are: block0, block1, block2")

    if args.load_session_dir is not None:
        load_session_dir = Path(args.load_session_dir).absolute()
        fixedLagCurrentSession = True
    else:
        load_session_dir = None
        fixedLagCurrentSession = False
        # fixedLagCurrentSession = True # FIXME: This is a temporary fix to allow the test to run without a prior session. The correct behavior should be to set this to False when no prior session is loaded.
    
    if args.save_session_dir is not None:
        save_session_dir = Path(args.save_session_dir).absolute()
    elif load_session_dir is not None:
        save_session_dir = load_session_dir.parent / f"{args.mission}_prior_{load_session_dir.name}"
    else:
        save_session_dir = moduleDir / "results/test_session"
        if save_session_dir.exists():
            import shutil
            shutil.rmtree(save_session_dir)

    # Resolve visualization flags: CLI --viz overrides module-level constants
    if args.viz is None:
        enable_mpl = ENABLE_MATPLOTLIB
        # enable_mpl = False
        enable_o3d = ENABLE_OPEN3D
        # enable_o3d = False
    else:
        enable_mpl = args.viz in ("mpl", "both")
        enable_o3d = args.viz in ("o3d", "both")

    params = Params(
        fixedLagCurrentSession=fixedLagCurrentSession,
        # fixedLagKeepFrames=50,
        # fixedLagMarg=FixedLagMarginalizationType.DENSE,
        rekeyPrior=True,
        loadSessionDir=load_session_dir,
        saveSessionDir=save_session_dir,
        bagFile=bag_file,
        gtFile=gt_file,
    )

    from dataclasses import asdict
    from pprint import pprint

    print("Running Multi-Session Test with the following parameters:")
    pprint(asdict(params), sort_dicts=False)
    print(f"Visualization: matplotlib={'on' if enable_mpl else 'off'}  open3d={'on' if enable_o3d else 'off'}")

    run_session(params, enable_matplotlib=enable_mpl, enable_open3d=enable_o3d)
