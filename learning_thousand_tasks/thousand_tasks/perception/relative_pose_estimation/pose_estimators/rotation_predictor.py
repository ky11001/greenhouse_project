from abc import abstractmethod

import torch
import torch.nn as nn

ModelRotPrediction = torch.Tensor
PredictedRotation = float


class RotationPredictorBase(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def load_weights(self):
        pass

    @abstractmethod
    def predict_rotation(self) -> ModelRotPrediction:
        pass

    @abstractmethod
    def prediction_to_radians(self, pred: ModelRotPrediction) -> PredictedRotation:
        pass

    @abstractmethod
    def forward(self, data: dict):
        pass
