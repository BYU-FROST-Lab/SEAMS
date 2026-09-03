import os
import math
from pathlib import Path
import pandas as pd
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

import gtsam


# =========================================================================
# GET LIDAR POINT CLOUDS FROM ROSBAG
# =========================================================================
class GroundTruthLoader:
    def __init__(self, path_to_gt: str | Path, dataset: str | None = None):
        if not os.path.exists(path_to_gt):
            raise FileNotFoundError(f"Ground truth file not found: {path_to_gt}")
        try:
            self.gt_df = pd.read_csv(path_to_gt, dtype=float)
        except Exception as e:
            try:
                self.gt_df = pd.read_csv(path_to_gt, delimiter=' ', dtype=float)
            except Exception as e2:
                try:
                    self.gt_df = pd.read_csv(path_to_gt, delimiter=', ', dtype=float)
                except Exception as e3:
                    raise ValueError(f"Failed to read ground truth file: {path_to_gt}. Error: {e3}") from e
        if self.gt_df.empty:
            raise ValueError(f"Ground truth file is empty: {path_to_gt}")

        self.dataset = dataset

        # self.gt_df.dropna(axis=1, how='all', inplace=True)

        if self.gt_df.shape[1] == 7:  # Assume data orientation is r, p, y
            # self._convert_euler_to_quat()
            self.isEuler = True
            self.gt_df.columns = [
                "t",
                "x", "y", "z",
                "roll", "pitch", "yaw"
            ]
            if isinstance(self.dataset, str):
                if self.dataset.lower() == "nclt":
                    self.gt_df["t"] = self.gt_df["t"] * 1e-6  # Convert to seconds from microseconds
        elif self.gt_df.shape[1] == 8:  # Assume data orientation is q_x, q_y, q_z, q_w
            self.isEuler = False
            self.gt_df.columns = [
                "t",
                "x", "y", "z",
                "q_x", "q_y", "q_z", "q_w",
            ]
        else:
            raise ValueError("Unexpected number of columns in ground truth data. Must be 7 (euler) or 8 (quat)")
        
        self.gt_arr = self.gt_df.values

    def _convert_euler_to_quat(self):
        """
        Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w).
        """
        # Assuming the columns are in the order: t, x, y, z, r, p, y
        # and we want to convert r, p, y to q_x, q_y, q_z, q_w
        tmp_df = pd.DataFrame(columns=["t", "x", "y", "z", "q_x", "q_y", "q_z", "q_w"])

        self.gt_df.columns = ["t", "x", "y", "z", "roll", "pitch", "yaw"]

        for i in tqdm(range(len(self.gt_df)), desc="Converting Euler to Quaternions"):
            roll = self.gt_df.iloc[i]['roll']
            pitch = self.gt_df.iloc[i]['pitch']
            yaw = self.gt_df.iloc[i]['yaw']

            # # Convert Euler angles to quaternion
            # cy = math.cos(yaw * 0.5)
            # sy = math.sin(yaw * 0.5)
            # cp = math.cos(pitch * 0.5)
            # sp = math.sin(pitch * 0.5)
            # cr = math.cos(roll * 0.5)
            # sr = math.sin(roll * 0.5)

            # quat = [
            #     cy * cp * sr - sy * sp * cr,  # q_x
            #     sy * cp * sr + cy * sp * cr,  # q_y
            #     sy * cp * cr - cy * sp * sr,  # q_z
            #     cy * cp * cr + sy * sp * sr   # q_w
            # ]

            quat = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()

            # assert np.allclose(quat, quat_test), f"Quaternions do not match: {quat} vs {quat_test}"

            tmp_df.loc[i] = [
                self.gt_df.iloc[i]['t'],  # t
                self.gt_df.iloc[i]['x'],  # x
                self.gt_df.iloc[i]['y'],  # y
                self.gt_df.iloc[i]['z'],  # z
                quat[0],  # q_x
                quat[1],  # q_y
                quat[2],  # q_z
                quat[3],  # q_w
            ]

        self.gt_df = tmp_df

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
        if self.isEuler:
            # Convert Euler angles to quaternion
            if self.dataset and self.dataset.lower():
                if self.dataset.lower() == "nclt":
                    # For NCLT, the Euler angles are in a different order
                    quat = R.from_euler('ZYX', [row[6], row[5], row[4]]).as_quat()
                else:
                    quat = R.from_euler('xyz', [row[4], row[5], row[6]]).as_quat()
            else:
                quat = R.from_euler('xyz', [row[4], row[5], row[6]]).as_quat()
            row = np.append(row[:4], quat)
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
    def is_euler(self):
        """Return whether the ground truth data is in Euler angle format."""
        return self.isEuler

    @property
    def dataframe(self):
        """Expose the ground truth dataframe for external use."""
        return self.gt_df
    
    @property
    def array(self):
        """Expose the ground truth array for external use."""
        return self.gt_arr



if __name__ == "__main__":

    file = Path("/home/benb/data_sets/NCLT/2012-01-08/groundtruth_2012-01-08.csv")

    gt = GroundTruthLoader(file)