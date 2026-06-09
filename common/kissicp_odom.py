import gtsam
from kiss_icp.config import load_config
from kiss_icp.kiss_icp import KissICP
from kiss_icp.datasets import dataset_factory
from pathlib import Path
import numpy as np

class KissICPOdom:
    def __init__(self, 
                 lidar_scan_path:str|Path, 
                 config_file_path:str|Path|None=None, 
                 max_range:float|None=None,
                 debug:bool=False
                 ):
        '''
        Class designed to give relative odometry between the last pose and the next pose.

        :params lidar_scan_path: path to directory where lidar point clouds are located. I believe possible file types are [bin, pcd, ply, xyz, obj, ctm, off, stl]
        :params config_file_path: config file for KissICP. None defaults to default parameters
        :params max_range: max range of points held in local map during registration
        :params debug: If true, prints debug statements
        '''

        # Load lidar scans. The generic dataloader should convert most types of lidar point cloud
        # files into numpy arrays to be used with KissICP
        lidar_scan_path = Path(lidar_scan_path)
        self.dataset_ = dataset_factory(
            dataloader="generic",
            data_dir=lidar_scan_path
        )
        self.current_dataset_index_ = 0

        # Load config if available and set up KissICP odometry
        if isinstance(config_file_path, str):
            config_file_path = Path(config_file_path)
        config = load_config(config_file_path, max_range)

        self.odometry_ = KissICP(config)

        # Keep a list of all poses and relative poses
        self.poses_ = []
        self.relative_poses_ = []

        self.debug_ = debug

    def set_initial_pose(self, pose:np.ndarray|gtsam.Pose3):
        '''
        Sets the initial pose for the odometry.
        
        :params pose: Either a 4x4 numpy array in form of a transformation matrix or a gtsam.Pose3
        '''

        if isinstance(pose, gtsam.Pose3):
            pose = pose.matrix()
        
        self.odometry_.last_pose = pose
    
    def process_next_keyframe(self) -> tuple[gtsam.Pose3, gtsam.Pose3|None]:
        '''
        Process lidar frames found in 'lidar_scan_path' passed to constructor.
        Processes the next frame sequentially each time this function is called.
        '''
        # Load next lidar information from dataset
        frame, timestamps = self.dataset_[self.current_dataset_index_]

        # Register the frame with the local map
        processed_frame, source = self.odometry_.register_frame(frame, timestamps)

        # Get new pose and relative pose
        new_pose = self.odometry_.last_pose.copy()
        new_pose = gtsam.Pose3(new_pose)
        self.poses_.append(new_pose)
        if self.current_dataset_index_ > 0:
            relative_pose = self.odometry_.last_delta.copy()
            relative_pose  = gtsam.Pose3(relative_pose)
            self.relative_poses_.append(relative_pose)
            if self.debug_:
                print(f"Frame {self.current_dataset_index_}: relative pose computed")  
                print(f"Translation: {relative_pose.translation()}")  
                print(f"Rotation matrix:\n{relative_pose.rotation().matrix()}")
        else:
            relative_pose = None

        self.current_dataset_index_ += 1

        return new_pose, relative_pose
    
    @property
    def poses(self):
        return self.poses_
    
    @property
    def relative_poses(self):
        return self.relative_poses_
    
    @property
    def local_map(self):
        return self.odometry_.local_map