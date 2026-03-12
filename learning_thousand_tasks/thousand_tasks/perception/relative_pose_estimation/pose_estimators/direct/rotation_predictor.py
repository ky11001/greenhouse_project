from typing import TypedDict

import numpy as np
import torch
from torch.nn.parameter import Parameter

from thousand_tasks.core.globals import ASSETS_DIR
from thousand_tasks.perception.relative_pose_estimation.pose_estimators.rotation_predictor import RotationPredictorBase


class ProcessedData(TypedDict):
    pointcloud_0: np.ndarray
    pointcloud_1: np.ndarray


ModelRotPrediction = torch.Tensor
PredictedRotation = float
PointCloud = torch.Tensor


class RotationPredictor(RotationPredictorBase):

    def __init__(self) -> None:
        super().__init__()
        self.weights_dir: str = None

    def load_weights(self):
        print("\n[POSE ESTIMATOR] Loading pretrained weights ...")
        ckpt_path = ASSETS_DIR / self.weights_dir
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        own_state = self.state_dict()
        for name, param in state_dict.items():
            name = name.split(".", 1)[1]
            if isinstance(param, Parameter):
                param = param.data
            own_state[name].copy_(param)
        print("[POSE ESTIMATOR] Done.\n")

    def forward(self, data: ProcessedData):
        pcd0: PointCloud = torch.tensor(data["pointcloud_0"]).unsqueeze(0)
        pcd1: PointCloud = torch.tensor(data["pointcloud_1"]).unsqueeze(0)
        with torch.no_grad():
            model_pred = self.predict_rotation(pcd0, pcd1)
        predicted_rot = self.prediction_to_radians(model_pred)
        return predicted_rot
