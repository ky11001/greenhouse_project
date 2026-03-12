import numpy as np
import torch
import torch.nn as nn

from thousand_tasks.perception.relative_pose_estimation.pose_estimators.direct.pointnet2_utils import PointNetSetAbstractionMsg, \
    PointNetSetAbstraction
from thousand_tasks.perception.relative_pose_estimation.pose_estimators.direct.rotation_predictor import RotationPredictor

ModelRotPrediction = torch.Tensor
PredictedRotation = float
PointCloud = torch.Tensor


class PointNet2_Encoder_v2(nn.Module):
    def __init__(self, output_dim, data_dim=0):
        super(PointNet2_Encoder_v2, self).__init__()
        self.data_present = data_dim > 0
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], data_dim,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.mlp1 = nn.Sequential(
            nn.Linear(1027, 512),
            nn.InstanceNorm1d(512),
            nn.Dropout(0.2),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.InstanceNorm1d(256),
            nn.Dropout(0.2),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.InstanceNorm1d(output_dim),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, xyz):
        bs, _, _ = xyz.shape
        if self.data_present:
            data = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            data = None
        l1_xyz, l1_points = self.sa1(xyz, data)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l3_points = l3_points.view(bs, 1024)
        l3_xyz = l3_xyz.view(bs, 3)
        out = torch.cat((l3_xyz, l3_points), dim=1)

        out = self.mlp1(out)
        out = self.mlp2(out)
        out = self.mlp3(out)

        return out


class RotationRegressor(RotationPredictor):

    def __init__(self) -> None:
        super(RotationRegressor, self).__init__()

        self.encoder = PointNet2_Encoder_v2(output_dim=256, data_dim=6)
        self.fusion = nn.Sequential(nn.Linear(in_features=512, out_features=256, bias=True),
                                    nn.InstanceNorm1d(256),
                                    nn.LeakyReLU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(in_features=256, out_features=128, bias=True),
                                  nn.InstanceNorm1d(128),
                                  nn.LeakyReLU(inplace=True))
        self.mlp3 = nn.Sequential(nn.Linear(in_features=128, out_features=1, bias=True),
                                  nn.Identity(inplace=True))

        self.weights_dir = "pose_estimator.ckpt"

    def predict_rotation(self, pcd0: PointCloud, pcd1: PointCloud) -> ModelRotPrediction:
        encoded_live = self.encoder(pcd0)
        encoded_bottleneck = self.encoder(pcd1)

        out = torch.concat((encoded_live, encoded_bottleneck), dim=1)
        out = self.fusion(out)
        out = self.mlp2(out)
        out = self.mlp3(out)

        return out[:, 0].cpu()

    def prediction_to_radians(self, pred: ModelRotPrediction) -> PredictedRotation:
        rot_pred = pred * torch.pi / 4
        return rot_pred.cpu().numpy()
