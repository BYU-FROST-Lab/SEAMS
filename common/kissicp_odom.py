import gtsam
from kiss_icp.config import load_config
from kiss_icp.kiss_icp import KissICP
from kiss_icp.datasets import dataset_factory
from pathlib import Path
import numpy as np

def _orthonormalize_se3(T: np.ndarray) -> np.ndarray:
    """
    Project the 3x3 rotation of an SE(3) matrix onto SO(3) via SVD.

    GTSAM Pose3.matrix() can leave R@R.T-I residuals ~1e-9, which Sophus
    (used inside KissICP's C++ registration) treats as a fatal assert.
    """
    T = np.asarray(T, dtype=np.float64).copy()
    R = T[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0.0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    T[:3, :3] = R_ortho
    return T

class KissICPOdom:
    def __init__(self, 
                 lidar_scan_path:str|Path|None=None,
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
        
        if lidar_scan_path is not None:
            # Load lidar scans. The generic dataloader should convert most types of lidar point cloud
            # files into numpy arrays to be used with KissICP
            lidar_scan_path = Path(lidar_scan_path)
            self.dataset_ = dataset_factory(
                dataloader="generic",
                data_dir=lidar_scan_path
            )
        else:
            self.dataset_ = None

        # Load config if available and set up KissICP odometry
        if isinstance(config_file_path, str):
            config_file_path = Path(config_file_path)
        # config = load_config(config_file_path, max_range)
        config = load_config(config_file_path)

        self.odometry_ = KissICP(config)

        # Keep a list of all poses and relative poses
        self.poses_ = []
        self.relative_poses_ = []
        self.scan_count_ = 0

        self.debug_ = debug

    def set_initial_pose(self, pose:np.ndarray|gtsam.Pose3):
        '''
        Sets the initial pose for the odometry.
        
        :params pose: Either a 4x4 numpy array in form of a transformation matrix or a gtsam.Pose3
        '''

        if isinstance(pose, gtsam.Pose3):
            pose = pose.matrix()

        # GTSAM matrices can fail Sophus's strict orthogonality check (~1e-9 residual).
        pose = _orthonormalize_se3(pose)
        self.odometry_.last_pose = pose
        self.odometry_.last_delta = np.eye(4)  # Reset last delta to identity for the first frame
        
        self.poses_ = []
        self.relative_poses_ = []
        self.scan_count_ = 0
    
    def process_next_keyframe(self, xyz_points: np.ndarray|None = None) -> tuple[gtsam.Pose3, gtsam.Pose3|None, np.ndarray]:
        '''
        Process lidar frames found in 'lidar_scan_path' passed to constructor.
        Processes the next frame sequentially each time this function is called.
        '''
        if self.dataset_ is not None:
            # Load next lidar information from dataset
            frame, timestamps = self.dataset_[self.scan_count_]

            # Register the frame with the local map
            deskewed_points, source = self.odometry_.register_frame(frame, timestamps)
        elif xyz_points is not None:
            # Register the frame with the local map
            fake_timestamps = np.array([]) # Internal KissICP optimization tracking expects an empty array to disable deskewing
            deskewed_points, source = self.odometry_.register_frame(xyz_points, fake_timestamps)
        else:
            raise ValueError("No dataset or xyz_points provided for processing.")
        
        # Get new pose and relative pose
        new_pose = gtsam.Pose3(self.odometry_.last_pose.copy())
        self.poses_.append(new_pose)
        if self.scan_count_ > 0:
            relative_pose  = gtsam.Pose3(self.odometry_.last_delta.copy())
            self.relative_poses_.append(relative_pose)
            if self.debug_:
                print(f"Frame {self.scan_count_}: relative pose computed")
                print(f"Translation: {relative_pose.translation()}")  
                print(f"Rotation matrix:\n{relative_pose.rotation().matrix()}")
        else:
            relative_pose = None

        self.scan_count_ += 1

        return new_pose, relative_pose, deskewed_points
    
    @property
    def poses(self):
        return self.poses_
    
    @property
    def relative_poses(self):
        return self.relative_poses_
    
    @property
    def local_map(self):
        return self.odometry_.local_map
    

class KissICPOdomLive:
    def __init__(self, config_file_path: str | Path | None = None, max_range: float | None = None):
        """
        Modified KissICP tracker fed directly with raw byte points array buffers
        from real-world stream loops instead of file system paths.
        """
        if isinstance(config_file_path, str):
            config_file_path = Path(config_file_path)
            
        config = load_config(config_file_path, max_range=max_range)
        self.odometry_ = KissICP(config)
        
        # Absolute structural path integration storage arrays
        self.poses_ = []              # <-- Tracks the composed absolute gtsam.Pose3 path
        self.relative_poses_ = []     # Tracks local frame-to-frame delta transformations
        self.scan_count_ = 0

    def set_initial_pose(self, pose: np.ndarray | gtsam.Pose3):
        if isinstance(pose, gtsam.Pose3):
            pose = pose.matrix()
        # Same Sophus orthogonality guard as KissICPOdom (GTSAM matrices ~1e-9 residual).
        pose = _orthonormalize_se3(pose)
        self.odometry_.last_pose = pose
        
        # Prime the trajectory array stack with our starting base transform matrix
        self.poses_ = [gtsam.Pose3(pose)]
        self.relative_poses_ = []
        self.scan_count_ = 0
    
    def process_next_keyframe(self, xyz_points: np.ndarray) -> tuple[gtsam.Pose3, gtsam.Pose3 | None]:
        """
        Processes a raw live structured point cloud array matrix with explicit 
        geometric singularity protection gates.
        """
        # Internal KissICP optimization tracking expects an empty array to disable deskewing
        fake_timestamps = np.array([]) 

        # =====================================================================
        # HARDENED SINGULARITY GUARD: FORCE INITIAL IDENTITY PASSTHROUGH
        # =====================================================================
        if self.scan_count_ == 0:
            try:
                self.odometry_.last_delta = np.eye(4)
                _, _ = self.odometry_.register_frame(xyz_points, fake_timestamps)
            except Exception as e:
                pass
                
            self.odometry_.last_pose = np.eye(4) if self.odometry_.last_pose is None else self.odometry_.last_pose
            self.odometry_.last_delta = np.eye(4)
            
            # Capture the base composed state
            last_pose = gtsam.Pose3(self.odometry_.last_pose)
            self.poses_.append(last_pose)
            self.scan_count_ += 1
            return last_pose, None
        # =====================================================================

        # Standard pipeline pass for all subsequent frames (where voxel map is populated)
        processed_frame, source = self.odometry_.register_frame(xyz_points, fake_timestamps)

        # 1. Capture the new COMPOSED global pose accumulated by KissICP
        composed_pose = gtsam.Pose3(self.odometry_.last_pose.copy())
        self.poses_.append(composed_pose)
        
        # 2. Capture the relative frame-to-frame step change
        relative_pose = gtsam.Pose3(self.odometry_.last_delta.copy())
        self.relative_poses_.append(relative_pose)

        self.scan_count_ += 1
        
        # Return the absolute composed pose along with the relative step delta
        return composed_pose, relative_pose

    @property
    def poses(self):
        """Exposes the absolute composed trajectory history log."""
        return self.poses_
        
    @property
    def relative_poses(self):
        """Exposes the raw frame-to-frame delta transformation steps."""
        return self.relative_poses_
    
    @property
    def local_map(self):
        return self.odometry_.local_map