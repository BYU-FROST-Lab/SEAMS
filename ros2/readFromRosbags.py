import os
import sys
import math
from pathlib import Path
import argparse
import time
import pandas as pd

from rosbags.highlevel import AnyReader
from matplotlib import pyplot as plt
import numpy as np

import gtsam


# FORCE X11 compatibility mode for Wayland sessions
os.environ["XDG_SESSION_TYPE"] = "x11"

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from .jitter_buffer import JitterBuffer, BufferedMessage

# Grab the standard ROS 2 type definitions
from rosbags.typesys import Stores, get_typestore
typestore = get_typestore(Stores.ROS2_HUMBLE)




# =========================================================================
# GET LIDAR POINT CLOUDS FROM ROSBAG
# =========================================================================
class GroundTruthLoader:
    def __init__(self, path_to_gt: str | Path):
        if not os.path.exists(path_to_gt):
            raise FileNotFoundError(f"Ground truth file not found: {path_to_gt}")
        self.gt_df = pd.read_csv(path_to_gt, delimiter=" ")
        self.gt_df.columns = [
            "t",
            "x", "y", "z",
            "q_x", "q_y", "q_z", "q_w",
        ]
        self.gt_arr = self.gt_df[[
            "t",
            "x", "y", "z",
            "q_x", "q_y", "q_z", "q_w",
        ]].values

    def get_closest_gt(self, timestamp: float):
        """
        Get the closest ground truth entry for a given timestamp.
        """
        if self.gt_arr.size == 0:
            raise ValueError("Ground truth array is empty. Ensure the ground truth file is loaded correctly.")
        gt_times = self.gt_arr[:, 0]
        closest_idx = np.argmin(np.abs(gt_times - timestamp))
        gt_row = self.gt_arr[closest_idx]

        gt_dt = abs(gt_row[0] - timestamp)
        print(f"Using closest GT at t={gt_row[0]:.6f}, bag t={timestamp:.6f}, dt={gt_dt:.3f}s")

        return self._tum_row_to_pose3(gt_row)
    
    def _tum_row_to_pose3(self, row: np.ndarray) -> gtsam.Pose3:
        """
        Convert a TUM format row to a GTSAM Pose3 object.
        """
        if len(row) != 8:
            raise ValueError("TUM row must have 8 elements: [t, x, y, z, q_x, q_y, q_z, q_w]")
        
        return gtsam.Pose3(
            gtsam.Rot3(
                row[7], # q_w
                row[4], # q_x
                row[5], # q_y
                row[6], # q_z
            ),
            gtsam.Point3(
                row[1], # x
                row[2], # y
                row[3], # z
            ),
        )
    
    def get_gt_as_values(self, symbol_prefix: str = "g") -> gtsam.Values:
        """
        Convert the entire ground truth dataframe to a GTSAM Values object.
        """
        values = gtsam.Values()
        for i, row in enumerate(self.gt_arr):
            pose = self._tum_row_to_pose3(row)
            key = gtsam.Symbol(symbol_prefix, i).key()
            values.insert(key, pose)
        return values
    
    @property
    def dataframe(self):
        """Expose the ground truth dataframe for external use."""
        return self.gt_df
    
    @property
    def array(self):
        """Expose the ground truth array for external use."""
        return self.gt_arr
    

# =========================================================================
# GET LIDAR POINT CLOUDS FROM ROSBAG
# =========================================================================
class ReadTopicsFromRosbag:
    def __init__(self, path_to_bag: str | Path, waiting_threshold=0.0, RUN_FULL_SIMULATION=True, simulation_length_s=60.0, use_jitter_buffer=False):
        '''
        This class takes a rosbag path and ground truth path (tum format) and runs a live odometry tracker on the bag data.
        '''

        self.lidar_topics = {
            "/os_cloud_node/points",
            "/ouster_top/points",
            "/ouster/points",
        }

        self.valid_topics = {
            *self.lidar_topics
        }
        
        self.buffer = JitterBuffer()
        self.exit_flag = False 
        self.filter_started = False 
        self.first_msg_timestamp = None 
        self.first_heading = None
        self.path_to_bag = path_to_bag
        self.waiting_threshold = waiting_threshold
        self.RUN_FULL_SIMULATION = RUN_FULL_SIMULATION
        self.simulation_length_s = simulation_length_s
        self.use_jitter_buffer = use_jitter_buffer

        self.increment_counter = 60
        self.increment_counter_step = self.increment_counter

        self.T_body_lidar = gtsam.Pose3(
                    gtsam.Rot3.RzRyRx(0.0, 0.0, np.pi),
                    gtsam.Point3(0.0, 0.0, 0.0),
                )
        

    def parse_lidar_measurements(self, connection, msg):

        # 1. Unpack raw point cloud array bytes from the rosbag message
        num_points = msg.width * msg.height
        raw_data = np.frombuffer(msg.data, dtype=np.uint8)
        point_bytes = raw_data.reshape(num_points, msg.point_step).copy()
        xyz_bytes = point_bytes[:, 0:12].copy()
        xyz_points = xyz_bytes.view(dtype=np.float32).reshape(num_points, 3)
        
        # R_body_lidar = self.T_body_lidar.rotation().matrix()
        # xyz_points = xyz_points @ R_body_lidar.T

        # 2. Filter out non-finite (NaN / Inf) returns
        clean_mask = np.isfinite(xyz_points).all(axis=1)
        xyz_clean = xyz_points[clean_mask]

        # 3. Dense Point Gate: Protect voxel allocation scaling
        if xyz_clean.shape[0] < 50:
            return np.empty((0, 3), dtype=np.float32)  # Return an empty array if too few points

        return xyz_clean

    def next_message(self):
        with AnyReader([Path(self.path_to_bag)], default_typestore=typestore) as reader:
            connections = [x for x in reader.connections]

            for connection, _t, rawdata in reader.messages(connections=connections):
                if connection.topic not in self.valid_topics:
                    continue
                
                try:
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    timestamp_ns = (msg.header.stamp.sec * 1_000_000_000) + msg.header.stamp.nanosec
                except KeyboardInterrupt:
                    self.exit_flag = True
                    break
                except: 
                    continue
                
                if not self.filter_started:
                    if self.first_msg_timestamp is None:
                        self.first_msg_timestamp = timestamp
                    if timestamp - self.first_msg_timestamp < self.waiting_threshold: # If I don't want to use the start of the bag for some reason, this is where you can tell it to wait
                        continue 
                    
                    self.first_msg_timestamp = timestamp
                    start_ts_ns = timestamp_ns
                    self.filter_started = True

                    print("🚀 Data stream detected. Initializing framework estimators")

                if timestamp_ns < start_ts_ns:
                    continue  
                if self.use_jitter_buffer:
                    self.buffer.push(timestamp_ns, connection, rawdata) 

                for ready_msg in self.buffer.pop_ready() if self.use_jitter_buffer else [BufferedMessage(timestamp_ns, connection, rawdata)]:
                    try:
                        connection = ready_msg.connection
                        timestamp = ready_msg.timestamp * 1e-9
                        msg = reader.deserialize(ready_msg.data, ready_msg.connection.msgtype)
                        if connection.topic in self.lidar_topics:
                            xyz_points = self.parse_lidar_measurements(connection, msg)
                        else:
                            continue

                    except KeyboardInterrupt:
                        self.exit_flag = True
                        break
                    except Exception as e:
                        print(f"Warning: Failed to process frame context: {e}")
                        break

                    if self.first_msg_timestamp is not None:
                        if timestamp - self.first_msg_timestamp > self.simulation_length_s and not self.RUN_FULL_SIMULATION:
                            self.exit_flag = True
                            print(f"Simulation length of {self.simulation_length_s} seconds reached. Exiting loop.")
                        if timestamp - self.first_msg_timestamp > self.increment_counter:
                            print(f"Reached {self.increment_counter} seconds in data stream loop.")
                            self.increment_counter += self.increment_counter_step
                    yield connection.topic, timestamp, xyz_points
                if self.exit_flag:
                    break

        for final_msg in self.buffer.flush(): 
            connection = final_msg.connection
            timestamp = final_msg.timestamp * 1e-9
            msg = reader.deserialize(final_msg.data, final_msg.connection.msgtype)
            if connection.topic in self.lidar_topics:
                xyz_points = self.parse_lidar_measurements(connection, msg)
                yield connection.topic, timestamp, xyz_points
            else:
                continue
        
        return

def main():
    parser = argparse.ArgumentParser(description="Run SLAM trajectory mission loops across explicit rosbags.")
    parser.add_argument("--run_full_simulation", action="store_false", help="Run full simulation parameters.")
    parser.add_argument("--simulation_length_s", type=float, default=60.0, help="Run simulation length clip duration.")
    args = parser.parse_args()


    mission = ReadTopicsFromRosbag("/home/benb/data_sets/NewerCollege/ros2/quad-easy")

    from common.kissicp_odom import KissICPOdomLive
    kiss_odom = KissICPOdomLive(max_range=150.0)
    
    for topic, timestamp, xyz_points in mission.next_message():
        kiss_odom.process_next_keyframe(xyz_points)
        pass

    points = []
    poses = kiss_odom.poses
    for pose in poses:
        points.append(pose.translation())
    
    # Plotting the trajectory
    points = np.array(points)
    plt.figure(figsize=(10, 6))
    plt.plot(points[:, 0], points[:, 1], marker='o', markersize=2, label='Trajectory')
    plt.title('KISS-ICP Trajectory')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()