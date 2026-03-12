from thousand_tasks.perception.relative_pose_estimation.pose_estimators.direct.rotation_regressor import RotationRegressor
from thousand_tasks.perception.relative_pose_estimation.pose_estimators.pose_predictor import PosePredictor


class PointentPoseRegressor(PosePredictor):

    def __init__(self,
                 device: str = "cuda:0",
                 filter_pointcloud: bool = True,
                 filter_outliers_o3d: bool = False,
                 n_points: int = 2048) -> None:
        super().__init__(device, filter_pointcloud, filter_outliers_o3d, n_points)
        self.rot_model = RotationRegressor()
        self._initialise_models()
