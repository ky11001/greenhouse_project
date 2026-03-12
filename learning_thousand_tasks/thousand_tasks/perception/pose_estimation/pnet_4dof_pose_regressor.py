import time

import numpy as np
import torch

from thousand_tasks.core.utils.scene_state import SceneState
from thousand_tasks.core.base_classes.rgbd_pose_estimation_base import RgbdPoseEstimationBase
from thousand_tasks.perception.relative_pose_estimation.pose_estimators.pnet_pose_estimators import PointentPoseRegressor
from thousand_tasks.core.utils.visualisation import draw_registration_result


class PointnetPoseRegressor_4dof(RgbdPoseEstimationBase):

    def __init__(self,
                 filter_pointcloud: bool = True,
                 filter_outliers_o3d: bool = False,
                 n_points: int = 2048,
                 T_WC: np.ndarray = None,
                 depth_units='mm',
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = torch.device('cuda'),
                 T_WC_demo=None,
                 **kwargs):

        self.pose_estimator = PointentPoseRegressor(device, filter_pointcloud, filter_outliers_o3d, n_points)
        self.extrinsics = T_WC
        self.extrinsics_demo = T_WC_demo if T_WC_demo is not None else T_WC

        super().__init__(depth_units=depth_units)

    def estimate_relative_pose(self,
                               scene1_state: SceneState,
                               scene2_state: SceneState,
                               visualise_pcds=False,
                               verbose=True,
                               **kwargs) -> np.ndarray:

        self.check_estimate_relative_pose_inputs(scene1_state, scene2_state)

        data = {
            "image_0": scene1_state.rgb.copy(),
            "image_1": scene2_state.rgb.copy(),
            "depth_0": scene1_state.depth.copy(),
            "depth_1": scene2_state.depth.copy(),
            "seg_0": scene1_state.segmap.copy(),
            "seg_1": scene2_state.segmap.copy(),
            "intrinsics_0": scene1_state.intrinsic_matrix.copy(),
            "intrinsics_1": scene2_state.intrinsic_matrix.copy(),
            "T_WC_0": self.extrinsics_demo,
            "T_WC_1": self.extrinsics
        }

        if verbose:
            print('Directly estimating pose...')
            start = time.time()

        T_delta_base, T_delta_cam = self.pose_estimator(data)

        if verbose:
            print(
                f"Estimation took {time.time() - start:.2f} s.")

        if visualise_pcds:
            pcd1_o3d = scene1_state.o3d_pcd
            pcd2_o3d = scene2_state.o3d_pcd
            draw_registration_result(source=pcd1_o3d, target=pcd2_o3d, transformation=T_delta_cam)

        return T_delta_base
