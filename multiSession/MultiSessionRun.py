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
from concurrent.futures import ThreadPoolExecutor, as_completed


# Local imports
import gtsam
import common.fgHelpers as fgtools
from multiSession.SessionManager import SessionManager
from params import Params, LoopClosure
from common.customTypes import DecisionType, FixedLagMarginalizationType
from common.LiveMapVisualizer import LiveMapVisualizer
from common.kissicp_odom import KissICPOdom
from params.params import Params
from rosBagRead.readFromRosbags import ReadTopicsFromRosbag
from common.groundTruthLoader import GroundTruthLoader
from common.datasetPaths import getMissionPaths

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

def update_session_worker(i, cs, deskewed_points, relPose_tmp, timestamp):
    try:
        result = cs.Update(
            deskewed_points,
            relPose=copy.deepcopy(relPose_tmp),
            timestamp=copy.copy(timestamp),
        )
        return i, result, None

    except Exception as e:
        return i, None, e


def run_session(params: Params = Params(), 
                enable_matplotlib: bool = ENABLE_MATPLOTLIB, 
                enable_open3d: bool = ENABLE_OPEN3D,
                policy: DecisionType | None = None,
                evaluate_decision_types: list[DecisionType] = [],
                lastRun: bool = True,
                bagStartSeconds: float = 1.0,
                simulationLength: float | None = None
                ):
    '''
    :param simulationLength: (Optional) Only run this many seconds of the bag. Runs the full bag if None.
    '''
    #########################################################################################
    # Session Parameters
    #########################################################################################

    # MAX_COUNT = 200
    MAX_COUNT = 5
    FIXED_WINDOW = False
    SAVE_FULL_SESSION = True
    SAME_SESSION_COV_CHANGE_PLOT = False
    SAVE_COVARIANCE_OF_CURRENT_POSE = False
    SAVE_UPDATE_TIMING = False
    SHOW_PRIOR_VALUES = False

    show_mpl = enable_matplotlib
    show_o3d = enable_open3d

    ax = None
    plt.close("all")

    #########################################################################################
    vizIndex=0
    if evaluate_decision_types and (params.loadSessionDir is not None or params.priorUpdateMethod == 2):
        currentSession = []
        still_valid = [True] * len(evaluate_decision_types)
        # Cache of scans keyed by (mission, timestamp_us) so each policy's prior is only read from disk once
        shared_pcds: dict = {}
        for idx, dt in enumerate(evaluate_decision_types):
            if dt == policy:
                vizIndex = idx
            tmpParams = copy.deepcopy(params)
            tmpParams.policyType = dt
            if tmpParams.loadSessionDir is not None:
                if Path(tmpParams.loadSessionDir / dt.name).exists():
                    tmpParams.loadSessionDir = tmpParams.loadSessionDir / dt.name
            session = SessionManager(params=tmpParams)
            if params.loadSessionDir is not None:
                try:
                    session.LoadSession(sharedScans=shared_pcds)
                except Exception as e:
                    still_valid[idx] = False
                    print(f"Couldn't load session for {dt.name}.")
            if params.dataset == 'nclt':
                session.SetGpsReference(lat=42.293227, lon=-83.709657, alt=270)

            currentSession.append(session)
    else:
        if params.loadSessionDir is not None:
            if Path(params.loadSessionDir / params.policyType.name).exists():
                params.loadSessionDir = params.loadSessionDir / params.policyType.name
        currentSession = [SessionManager(params.loadSessionDir, params)]
        if params.dataset == 'nclt':
            currentSession[0].SetGpsReference(lat=42.293227, lon=-83.709657, alt=270)
        
        still_valid = [True]
# |
# v
    bag1 = ReadTopicsFromRosbag(str(params.bagFile), waiting_threshold=bagStartSeconds,
                                RUN_FULL_SIMULATION=simulationLength is None,
                                simulation_length_s=simulationLength if simulationLength is not None else 60.0)

    if params.gtFile is not None:
        gt = GroundTruthLoader(str(params.gtFile), dataset=params.dataset)
    else:
        gt = None
        
    kissOdom = KissICPOdom(config_file_path=params.kissICPConfigFile)

    firstPass = True
    cnt = 0
    totalCnt = 0

    poses = []

    sessionSymbols = {ord('c'): 'C0'}#,ord('A'): 'C1', ord('B'): 'C2', ord('C'): 'C3'}
    for i, c in enumerate(currentSession[0].priorSessionSymbols_):
        sessionSymbols[ord(c)] = f'C{i+1}'

    if show_mpl:
        # fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(10, 8))

    viz = None
    if show_o3d:
        viz = LiveMapVisualizer(currentSession[vizIndex], window_name="MultiSession — Live Map")
        viz.start()

    start_time = time.time()
    firstTimestamp = None
    foundFirstIntraLoop = False
    prevISAM2 = gtsam.ISAM2()
    sscp_dets: dict [int,list[float]] = {}
    whole_graph_info_dets: list[float] = []
    cov_current_pose_arr = []
    updateTimes = []
    gps_msg = []

    plotTitle = f"Final Graph - {params.dataset.upper()}: {params.mission} - {currentSession[vizIndex].params.policyType.name}"
# |
# v

    for topic, timestamp, msg_return in bag1.next_message():
        if firstTimestamp is None:
            firstTimestamp = timestamp

        if topic in bag1.gps_topics:
            if not msg_return:
                continue
            gps_msg: list = [timestamp] + msg_return
            continue

        
        if firstPass and gt is not None:
            gtPose = gt.get_closest_gt(timestamp)
            if params.dataset == "nclt" and params.mission == "2012-01-08_bin":
                # GT in body frame. Need Lidar in body frame
                # Transformation pulled from Table 4 in https://robots.engin.umich.edu/nclt/nclt.pdf
                T_lidar2body = gtsam.Pose3(gtsam.Rot3.Ypr(np.deg2rad(-90.703), np.deg2rad(0.166), np.deg2rad(0.807)), 
                                gtsam.Point3(0.002, -0.004, -0.957))
                gtPose = gtPose.compose(T_lidar2body)
                # gtPose = T_lidar2body.compose(gtPose)
            print(f"Initial pose is: \n{gtPose}")
            kissOdom.set_initial_pose(gtPose)
        xyz_points, point_times = msg_return
        composedPose_odom, relPose_odom, deskewed_points = kissOdom.process_next_keyframe(xyz_points, point_times) # Poses are returned in the odometry frame, not the robot's frame
        pointCloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(deskewed_points))

        if firstPass:
            assert relPose_odom is None, "Relative pose should be None for the first keyframe"
            update_time = time.time()
            if gt is not None:
                # gtPose = gt.get_closest_gt(timestamp)
                # print(f"Initial pose is: \n{gtPose}")
                for cs in currentSession:
                    sessionResult = cs.Update(pointCloud, gtPose, timestamp=timestamp, gpsData=gps_msg)
                # poses.append(gtPose)
            else:
                for cs in currentSession:
                    sessionResult = cs.Update(pointCloud, composedPose_odom, timestamp=timestamp, gpsData=gps_msg)
                # poses.append(composedPose_odom)
                gps_msg.clear()
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
            results = [SessionManager.UpdateResult()] * len(currentSession)
            for i, cs in enumerate(currentSession):
                if not still_valid[i]:
                    continue
                if evaluate_decision_types:
                    try:
                        results[i] = cs.Update(pointCloud, relPose=copy.deepcopy(relPose_tmp), timestamp=copy.copy(timestamp), gpsData=gps_msg)
                    except Exception as e:
                        print(f"Error updating session {cs.params.policyType.name}: {e}")
                        still_valid[i] = False
                else:
                    results[i] = cs.Update(pointCloud, relPose=copy.deepcopy(relPose_tmp), timestamp=copy.copy(timestamp), gpsData=gps_msg)

# |
# v
            # sessionResults = [None] * len(currentSession)

            # with ThreadPoolExecutor(max_workers=len(currentSession)) as executor:

            #     futures = [
            #         executor.submit(
            #             update_session_worker,
            #             i,
            #             cs,
            #             deskewed_points,
            #             relPose_tmp,
            #             timestamp,
            #         )
            #         for i, cs in enumerate(currentSession)
            #         if still_valid[i]
            #     ]

            #     for future in as_completed(futures):
            #         i, result, error = future.result()

            #         if error is not None:
            #             print(
            #                 f"Error updating session "
            #                 f"{currentSession[i].params.policyType.name}: {error}"
            #             )
            #             still_valid[i] = False
            #         else:
            #             sessionResults[i] = result

            update_time = time.time() - update_time
            lastPose_odom = composedPose_odom
            # print(f"Frame {totalCnt}: relative pose computed")
            # print(f"    Translation: {relPose_tmp.translation()}, Rotation: {relPose_tmp.rotation().rpy()}")

        gps_msg.clear()

        if totalCnt == 35:
            pass
# |
# v
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

#             refKeys = currentSession[vizIndex].GetKeys(currentSession[vizIndex].addToPriorSymbol)
#             currIsam = currentSession[vizIndex].isam2
#             for k in refKeys:
#                 if k in sscp_dets:
#                     sscp_dets[k].append(np.linalg.det(currIsam.marginalCovariance(k))) # Take determinant of translation part of covariance matrix
#                 else:
#                     sscp_dets[k] = [np.linalg.det(currIsam.marginalCovariance(k))] # Take determinant of translation part of covariance matrix
#             # print(f"Time to calc full det: {time.time() - start}")

# |
# v
            ####################################
            # Get Covariance of Boundary nodes #
            ####################################

            refKeys = currentSession[vizIndex].boundaryNodes
            currIsam = currentSession[vizIndex].isam2
            for k in refKeys:
                fgtools.AddValueToDictList(sscp_dets, k, np.linalg.det(currIsam.marginalCovariance(k)))

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
            cov = currentSession[vizIndex].smoother_.marginalCovariance(currentSession[vizIndex].lastOdomKey_)
            cov_current_pose_arr.append(cov)

        if SAVE_UPDATE_TIMING:
            updateTimes.append(update_time)

        # odom.update_last_pose(currentSession.values.atPose3(currentSession.values.keys()[-1]))
        # if currentSession[0].currTimestamp_ == 730:
        #     MAX_COUNT = 0
        if cnt >= MAX_COUNT: # or totalCnt > 225:
        # if True:
            cnt = 0
# |
# v
            if show_mpl:
                if ax:
                    ax.clear()
                else:
                    ax = fgtools.plot_graph(gtsam.NonlinearFactorGraph(), gtsam.Values())
                # fgtools.plot_graph(currentSession.prior.graph, currentSession.prior.values, ax=ax, color_override=bag_colors[0], alpha=0.5)
                # fgtools.plot_graph(currentSession.current.graph, currentSession.current.values, ax=ax, title="Testing Loop Closure in Graph", color_override=bag_colors[1], alpha=0.5)
                tmpValues = currentSession[vizIndex].values
                # fgtools.plot_graph(currentSession.priorGraph, currentSession.priorValues, color_override="tab:grey", ax=ax, title="Testing Combined Graph", changeColorBySymbols=False)
                # fgtools.plot_graph_covariances(currentSession.priorValues.keys(), currentSession.priorGraph, currentSession.priorValues, ax=ax, edgecolor="tab:grey")
                if gt is not None:
                    ax.plot(gt.array[:,1], gt.array[:,2], color='tab:green', label='Ground Truth', alpha=0.5)
                    # ax.plot(gt2.array[:,1], gt2.array[:,2], color='tab:orange', label='Ground Truth Block0', alpha=0.5)
                fgtools.plot_graph(currentSession[vizIndex].graph, tmpValues, ax=ax, title=plotTitle, changeColorBySymbols=True, addKeyLabels=False, symbolColors=sessionSymbols)
                if currentSession[vizIndex].params.policyType == DecisionType.RELATIVE_COVARIANCE:
                    fgtools.plot_graph_covariances(tmpValues.keys(), currentSession[vizIndex].fixedLagSmoother, tmpValues, ax=ax, changeColorBySymbols=True, symbolColors=sessionSymbols)
                # Plot prior tree keys
                if currentSession[vizIndex].interLoopTree_ is not None and SHOW_PRIOR_VALUES:
                    fgtools.plot_points(currentSession[vizIndex].interLoopTree_.keys, currentSession[vizIndex].currentEstimate_, "tab:red", ax=ax)
                # odom_values = gtsam.Values()
                # for i, pose in enumerate(kissOdom.poses):
                #     odom_values.insert(gtsam.Symbol('o', i).key(), pose)
                # fgtools.plot_points(odom_values.keys(), odom_values, color='tab:red', ax=ax)
                
                if FIXED_WINDOW:
                    windowKeys = currentSession[vizIndex].GetMostRecentKeys(params.fixedLagKeepFrames+1)
                    p1 = tmpValues.atPose3(windowKeys[-1])
                    p2 = tmpValues.atPose3(windowKeys[0])
                    offset = 5
                    ax.set_xlim(min(p1.x(), p2.x())-offset, max(p1.x(), p2.x())+offset)
                    ax.set_ylim(min(p1.y(), p2.y())-offset, max(p1.y(), p2.y())+offset)
#########
                plt.pause(0.05)
                pass
#########
            if show_o3d and not viz.update():
                print("Open3D window closed — disabling live 3D visualization.")
                viz.close()
                viz = None
                show_o3d = False

        cnt += 1
        totalCnt += 1
        prevISAM2 = currentSession[vizIndex].isam2

        if not any(still_valid):
            print("\n")
            raise RuntimeError("All policies failed. Check output")

    end_time = time.time()
    bag_time = timestamp - firstTimestamp
    run_time = end_time - start_time

    isRealTime = run_time < bag_time
    print(f"Real-time performance: {"\033[92m" if isRealTime else "\033[91m"}{isRealTime}\033[0m" \
            f" (bag time: {bag_time:.2f}s, run time: {run_time:.2f}s)")

    if show_o3d and viz is not None:
        viz.close()

    if show_mpl:
        plt.close(fig)

    for i, cs in enumerate(currentSession):
        if not still_valid[i]:
            print(f"Session {cs.params.policyType.name} encountered an error and was not completed.")
            continue
        ax = fgtools.plot_graph(gtsam.NonlinearFactorGraph(), gtsam.Values())
        
        plotTitle = f"Final Graph - {params.dataset.upper()}: {params.mission} - {cs.params.policyType.name}"

        final_graph, final_values, _, _ = cs.GetFinalSessionInfo()
        ax = fgtools.plot_graph(final_graph, final_values, ax=ax, title=plotTitle, addKeyLabels=False, changeColorBySymbols=True, symbolColors=sessionSymbols)
# ^
# |
        if SAVE_FULL_SESSION:
            cs.SaveSession(params.saveSessionDir)
            if params.loadSessionDir:
                directory = params.saveSessionDir / cs.params.policyType.name
            else:
                directory = Path(params.saveSessionDir)
            # Save matplotlib figure
            plt.savefig(directory / "final_graph.png", dpi=300, bbox_inches='tight')

            # Save figure with covariances
            try:
                fgtools.plot_graph_covariances(final_values.keys(), cs.smoother_, final_values, ax=ax, changeColorBySymbols=True, symbolColors=sessionSymbols)
                plt.savefig(directory / "final_graph_with_covariances.png", dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Error occurred while plotting covariances: {e}")
            ax.clear()

            # Plot the final graph and save with ground truth
            fgtools.plot_graph(final_graph, final_values, ax=ax, title=plotTitle, addKeyLabels=False, changeColorBySymbols=True, symbolColors=sessionSymbols)
            if gt is not None:
                ax.plot(gt.array[:,1], gt.array[:,2], color='tab:green', label='Ground Truth', alpha=0.5)
                plt.savefig(directory / "final_graph_with_ground_truth.png", dpi=300, bbox_inches='tight')
        else:
            Path.mkdir(directory, exist_ok=True)

    if SAME_SESSION_COV_CHANGE_PLOT:
        # Save dictionary for plotting elsewhere
        with open(params.saveSessionDir / "det_of_addToPriorNodes.pkl", "wb") as f:
            pickle.dump(sscp_dets, f)   

    if SAVE_COVARIANCE_OF_CURRENT_POSE:
        with open(params.saveSessionDir / "cov_of_currentPoses.pkl", "wb") as f:
            pickle.dump(cov_current_pose_arr, f)
# ^
# |
    if SAVE_UPDATE_TIMING:
        with open(params.saveSessionDir / "update_timing.pkl", "wb") as f:
                    pickle.dump(updateTimes, f)

    print("Done with session\n")
    
    if lastRun:
        print("Close plots to exit")
        plt.show()
        pass
    else:
        print("Only showing plots for 10 seconds")
        plt.pause(10)




def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Run Multi-Session Test")
    # Argument for which dataset to use (current option is to use the Newer College dataset)
    parser.add_argument("-d", "--dataset", type=str, default="nc", choices=["nc", "nclt", "urban"],help="Dataset to use (default: nc). Options: nc, nclt, urban")
    # Argument for which mission to run in the dataset
    parser.add_argument("-m", "--missions", nargs="+", type=str, default="", help="Mission to run in the dataset.")
    # Argument for the form of updating prior map
    parser.add_argument("--prior_update_method", type=int, default=None, help="How to update the prior map. 0: Post session, 1: incremental, or 2: continuous")
    # Argument for the load session directory (optional)
    parser.add_argument("-l", "--load_session_dir", type=str, default=None, help="Directory to load a previous session from (default: None)")
    # Argument for the save session directory (optional)
    parser.add_argument("-s", "--save_session_dir", type=str, default=None, help="Directory to save the current session to (default: None)")
    # Argument for the global scans directory (optional)
    parser.add_argument("--scans_dir", type=str, default=None, help=f"Global directory scans are saved to and loaded from as <scans_dir>/<dataset>/<mission>/<timestamp_us>.pcd (default: {Params().scansRootDir})")
    # Visualization mode
    parser.add_argument(
        "-v", "--viz",
        type=str,
        default=None,
        choices=["none", "mpl", "o3d", "both"],
        help="Visualization mode: none | mpl (matplotlib only) | o3d (Open3D only) | both. "
             "Defaults to the module-level ENABLE_MATPLOTLIB / ENABLE_OPEN3D flags.",
    )
    parser.add_argument_group("Decision Type Evaluation", "Options for evaluating different decision types")
    parser.add_argument(
        "-e",
        "--evaluate_decision_types",
        type=list[str],
        default=[],
        choices=[dt.name for dt in DecisionType],
        help="Evaluate different decision types by running the session multiple times with different decision policies. "
             "This will override the --policy argument and run the session for each decision type selected.",
    )
    parser.add_argument(
        "-p",
        "--policy",
        type=str,
        default=None,
        choices=[dt.name for dt in DecisionType],
        help="Decision policy to use for the session. Options: DISTANCE, OVERLAP, INLIER_RATIO, INLIER_RMSE, TWO_STEP_VALIDATION, RELATIVE_COVARIANCE. "
             "This argument is ignored if --evaluate_decision_types is set.",
    )
    parser.add_argument(
        "-t",
        "--transition_check",
        action="store_true",
        default=None,
        help="Used to make the boundary between localization and mapping well constrained."
        "Checks the determinant of first and last mapping nodes."
    )
    
    args = parser.parse_args()

    dirAfterResults = ""

    ########################################
    # Hardcode values for testing purposes #
    ########################################

    # args.evaluate_decision_types = [
    # # #     "NONE",
    #     # "DISTANCE",
    # # #     "OVERLAP",
    #     "INLIER_RATIO",
    #     # "INLIER_RMSE",
    # #     "TWO_STEP_VALIDATION",
    # #     # "TWO_OPTION",
    # # #     "RELATIVE_COVARIANCE",
    # ]   
    args.policy = "DISTANCE"
    # args.viz = "none"
    # args.viz = "o3d"
    args.prior_update_method = 0 # 0: post, 2: continuous
    args.transition_check = True
    dirAfterResults = "boundaryEval"

    # args.dataset = "nc"
    # args.missions = [
    #     # "quad-easy",
    #     # "quad-medium",
    #     "quad-hard",
    #     # "cloister",
    #     # "park",
    # ]
    # # args.load_session_dir = moduleDir / "results/accuracyEval/nc/wellConstrained/continuous/cloister__quad-hard__quad-medium__quad-easy/"
    # # args.load_session_dir = moduleDir / "results/nc/post/cloister__quad-hard__quad-medium__quad-easy"
    # args.load_session_dir = moduleDir / "results/accuracyEval/nc/exact/post/quad-easy"
    # args.save_session_dir = moduleDir / "results/tests/nc/exact/post/quad-hard__quad-easy"
    # args.load_session_dir = moduleDir / "results/twoStepEval/nc/exact/post/quad-easy"
    # args.load_session_dir = moduleDir / "results/twoStepEval2/nc/wellConstrained/post/cloister__quad-hard__quad-medium__quad-easy"
    # args.save_session_dir = moduleDir / "results/test/nc/wellConstrained/post/quad-medium__quad-easy"
    # args.load_session_dir = moduleDir / "results/test/nc/wellConstrained/post/quad-medium__quad-easy"
    # args.load_session_dir = moduleDir / "results/accuracyEval/nc/wellConstrained/post/cloister__quad-hard__quad-medium__quad-easy"
    # args.save_session_dir = moduleDir / "results/transitionBoundaryEval/nc/exact/quad-hard__quad-easy"
    # args.save_session_dir = moduleDir / "results/transitionBoundaryEval/nc/wellConstrained/quad-hard__quad-easy"
    # # args.save_session_dir = moduleDir / "results/nc/well_constrained/quad-medium__quad-easy"
    # # args.save_session_dir = moduleDir / "results/marginalizationEvaluation/nc/approx__quad-hard__quad-easy"
    # args.save_session_dir = moduleDir / "results/accuracyEval/nc/exact/continuous/quad-easy/"

    args.dataset = "urban"
    args.missions = [
        # "block0",
        "block1",
        # "block2",
    ]
    # args.load_session_dir = None
    # args.load_session_dir = moduleDir / "results/urban/block0/"
    # args.save_session_dir = moduleDir / "results/urban/block0"
    args.load_session_dir = moduleDir / "results/boundaryEval/urban/exact/post/block0/"
    args.save_session_dir = moduleDir / "results/boundaryEvalBadLoop/urban/exact/post/block1__block0/"
    # args.load_session_dir = moduleDir / "policy_evaluation/urban2/block1_prior_block0/"
    # args.load_session_dir = moduleDir / "policy_evaluation/urban2/block0"
    # args.viz = "none"

    # args.dataset = "nclt"
    # args.missions = [
    # #     # "2012-01-08",
    #     "2012-01-15",
    #     # "2012-01-22",
    #     # "2012-04-29",
    #     # "2012-05-11",
    #     # "2012-06-15",
    #     # "2013-02-23",
    # ]
    # args.load_session_dir = moduleDir / "results/nclt/post/2012-01-08"
    # args.save_session_dir = moduleDir / "results/twoStepVal3/nclt/exact/post/2012-01-15__2012-01-08"
    # # # # args.save_session_dir = moduleDir / "results/nclt/1_nclt2ros"

    ########################################

    priorUpdateMethod = Params().priorUpdateMethod if args.prior_update_method is None else args.prior_update_method
    priorUpdateMethodString = "post" if priorUpdateMethod == 0 else "incremental" if priorUpdateMethod == 1 else "continuous"
    transitionBroundaryCheck = Params().enableTransitionInformationCheck if args.transition_check is None else args.transition_check
    transitionBroundaryTypeString = "wellConstrained" if transitionBroundaryCheck else "exact"

    i = 0
    for i, mission in enumerate(args.missions):

        bag_file, gt_file = getMissionPaths(args.dataset, mission)

        if not bag_file.exists():
            print(f"Bag file {bag_file} does not exist")
        if not gt_file.exists():
                print(f"Ground truth file {gt_file} does not exist")


        if i == 0:
            if args.load_session_dir is not None:
                load_session_dir = Path(args.load_session_dir).absolute()
            else:
                load_session_dir = None
        else:
            load_session_dir = copy.copy(save_session_dir)
            args.save_session_dir = None
        
        if args.save_session_dir is not None:
            save_session_dir = Path(args.save_session_dir).absolute()
        elif load_session_dir is not None:
            save_session_dir = load_session_dir.parent / f"{mission}__{load_session_dir.name}"
        else:
            if args.dataset and mission:
                save_session_dir = moduleDir / "results" / dirAfterResults /args.dataset / transitionBroundaryTypeString / priorUpdateMethodString / mission
            else:
                save_session_dir = moduleDir / "results/test_session"
            while save_session_dir.exists():
                print(f"Warning: save_session_dir already exists {save_session_dir}")
                parts = str(save_session_dir).strip().rsplit('_', 1)
                if len(parts) > 1 and parts[-1].isdigit():
                    save_session_dir = f"{parts[0]}_{int(parts[1])+1}"
                else:
                    save_session_dir = f"{save_session_dir}_2"
                save_session_dir = Path(save_session_dir)


        # Resolve visualization flags: CLI --viz overrides module-level constants
        if args.viz is None:
            enable_mpl = ENABLE_MATPLOTLIB
            # enable_mpl = False
            enable_o3d = ENABLE_OPEN3D
            # enable_o3d = False
        else:
            enable_mpl = args.viz in ("mpl", "both")
            enable_o3d = args.viz in ("o3d", "both")

        policy = Params().policyType if args.policy is None else DecisionType[args.policy.upper()]
        decisionTypes = [DecisionType[p.upper()] for p in args.evaluate_decision_types]

        if args.dataset.lower() == "nclt" and not load_session_dir:
            maxCorrespondingDistance = 5.0
        else:
            maxCorrespondingDistance = 0.5

        boundaryThresholds = [0.01, 0.05, 0.25, 0.5, 1.0, 2.0]
        # boundaryThresholds = [np.inf]
        # boundaryThresholds = [0.1]
        for boundaryThreshold in boundaryThresholds:
            params = Params(
                policyType=policy,
                rekeyPrior=True,
                loadSessionDir=load_session_dir,
                saveSessionDir=save_session_dir.parent / f"Loc2Map_{int(boundaryThreshold*100) if np.isfinite(boundaryThreshold) else "inf"}" / save_session_dir.name,
                dataset=args.dataset.lower(),
                mission=mission,
                bagFile=bag_file,
                gtFile=gt_file,
                priorUpdateMethod=priorUpdateMethod,
                enableTransitionInformationCheck=transitionBroundaryCheck,
                scansRootDir=Path(args.scans_dir).absolute() if args.scans_dir is not None else Params().scansRootDir,
                loopClosure=LoopClosure(
                    maxCorrespondenceDist=maxCorrespondingDistance
                ),
                boundaryLoc2Map=boundaryThreshold,
            )

            from dataclasses import asdict
            from pprint import pprint

            print("Running Multi-Session Test with the following parameters:")
            pprint(asdict(params), sort_dicts=False)
            print(f"Visualization: matplotlib={'on' if enable_mpl else 'off'}  open3d={'on' if enable_o3d else 'off'}")

            run_session(params, 
                        enable_matplotlib=enable_mpl, 
                        enable_open3d=enable_o3d, 
                        policy=policy, 
                        evaluate_decision_types=decisionTypes,
                        lastRun=True if i == len(args.missions)-1 and boundaryThreshold == boundaryThresholds[-1] else False,
                        bagStartSeconds=30.0,
                        simulationLength=100.0
                        )


if __name__ == "__main__":
    main()