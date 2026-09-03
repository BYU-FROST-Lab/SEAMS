import os
import sys
from pathlib import Path
import argparse

from rosbags.highlevel import AnyReader
from matplotlib import pyplot as plt
import numpy as np
import yaml

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
class ReadTopicsFromRosbag:
    def __init__(self, path_to_bag: str | Path, waiting_threshold=0.0, RUN_FULL_SIMULATION=True, simulation_length_s=60.0, use_jitter_buffer=False):
        '''
        This class takes a rosbag path and ground truth path (tum format) and runs a live odometry tracker on the bag data.
        '''

        self.lidar_topics = {
            "/os_cloud_node/points",    # NC - 2021
            "/os1_cloud_node/points",   # NC - 2020
            "/ouster_top/points",       # BYU
            "/ouster/points",           # Ouster
            "/velodyne_points",         # NCLT
        }

        self.gps_topics = {
            "/navsat/fix",                      # NCLT
            "/inertial_sense/gps1/NavSatFix",   # BYU
        }

        self.valid_topics = {
            *self.lidar_topics,
            *self.gps_topics
        }
        
        self.buffer = JitterBuffer()
        self.exit_flag = False 
        self.filter_started = False 
        self.first_msg_timestamp = None 
        self.first_heading = None
        self.path_to_bag = Path(path_to_bag)
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

    def parse_gps_measurements(self, connection, msg):
        '''
        Return list[lat, lon, alt, cov, cov_type]
        '''
        if msg.status.status < 0:
            return []
        lat = msg.latitude
        lon = msg.longitude
        alt = msg.altitude
        if not np.isfinite(alt):
            return []
        cov = np.asarray(msg.position_covariance).reshape(3, 3)
        cov_type = msg.position_covariance_type

        return [lat, lon, alt, cov, cov_type]

    def next_message(self):
        '''
        Return (topic, timestamp, msg_return)
        msg_return:
           lidar: xyz_points (Nx3)
           gps: lat, long, alt, cov, cov_type as list of len 5 
        '''
        # Determine total rosbag duration once
        if not hasattr(self, 'bag_duration') or self.bag_duration is None:
            self.get_bag_time_range()

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
                            msg_return = self.parse_lidar_measurements(connection, msg)
                        elif connection.topic in self.gps_topics:
                            msg_return = self.parse_gps_measurements(connection, msg)
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

                            elapsed_time = timestamp - self.bag_start_timestamp

                            completion_percentage = (
                                elapsed_time / self.bag_duration
                            ) * 100.0

                            completion_percentage = min(100.0, max(0.0, completion_percentage))

                            print(
                                f"Reached {self.increment_counter} seconds in data stream loop. "
                                f"Completion: {completion_percentage:.1f}%"
                            )

                            self.increment_counter += self.increment_counter_step
                    # msg_return:
                    #   xyz_points for lidar (Nx3)
                    #   lat, long, alt, cov, cov_type as list of len 5
                    yield connection.topic, timestamp, msg_return
                if self.exit_flag:
                    break

        for final_msg in self.buffer.flush(): 
            connection = final_msg.connection
            timestamp = final_msg.timestamp * 1e-9
            msg = reader.deserialize(final_msg.data, final_msg.connection.msgtype)
            if connection.topic in self.lidar_topics:
                msg_return = self.parse_lidar_measurements(connection, msg)
            elif connection.topic in self.gps_topics:
                msg_return = self.parse_gps_measurements(connection, msg)
            else:
                continue
            yield connection.topic, timestamp, msg_return
        
        return

    def get_bag_time_range(self):
        """Get the start and end timestamps of the rosbag."""

        print("Calculating bag duration...")

        with open(self.path_to_bag / "metadata.yaml", "r") as f:
            metadata = yaml.safe_load(f)

        self.bag_duration = metadata["rosbag2_bagfile_information"]["duration"]["nanoseconds"] * 1e-9

        self.bag_start_timestamp = metadata["rosbag2_bagfile_information"]["starting_time"]["nanoseconds_since_epoch"] * 1e-9

        print(
            f"Rosbag duration: {self.bag_duration:.2f} seconds "
            f"({self.bag_duration / 60.0:.2f} minutes)"
        )
        # with AnyReader([Path(self.path_to_bag)], default_typestore=typestore) as reader:
        #     start_time = None
        #     end_time = None

        #     for connection, timestamp, _rawdata in reader.messages():
        #         if connection.topic not in self.valid_topics:
        #             continue

        #         if start_time is None:
        #             start_time = timestamp

        #         end_time = timestamp

        # if start_time is None or end_time is None:
        #     raise RuntimeError("No valid LiDAR messages found in rosbag.")

        # self.bag_start_timestamp = start_time * 1e-9
        # self.bag_end_timestamp = end_time * 1e-9
        # self.bag_duration = (
        #     self.bag_end_timestamp - self.bag_start_timestamp
        # )

        # print(
        #     f"Rosbag duration: {self.bag_duration:.2f} seconds "
        #     f"({self.bag_duration / 60.0:.2f} minutes)"
        # )

def main():
    parser = argparse.ArgumentParser(description="Run SLAM trajectory mission loops across explicit rosbags.")
    parser.add_argument("--run_full_simulation", action="store_false", help="Run full simulation parameters.")
    parser.add_argument("--simulation_length_s", type=float, default=60.0, help="Run simulation length clip duration.")
    args = parser.parse_args()


    mission = ReadTopicsFromRosbag("/home/benb/data_byu/bags/urban/block0/")

    from common.kissicp_odom import KissICPOdomLive
    kiss_odom = KissICPOdomLive(max_range=150.0)
    
    for topic, timestamp, msg_return in mission.next_message():
        if topic in mission.lidar_topics:
            kiss_odom.process_next_keyframe(msg_return)
        elif topic in mission.gps_topics:
            print(f"{timestamp}: {msg_return}")
        else:
            continue
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