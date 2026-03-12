from abc import ABC, abstractmethod

import numpy as np

from thousand_tasks.core.utils.scene_state import SceneState


class RgbdPoseEstimationBase(ABC):

    def __init__(self, depth_units='mm'):
        assert depth_units in ['mm', 'm']
        self.depth_units = depth_units

    @abstractmethod
    def estimate_relative_pose(self,
                               scene1_state: SceneState,
                               scene2_state: SceneState,
                               **kwargs) -> np.ndarray:
        pass

    def check_estimate_relative_pose_inputs(self, scene1_state: SceneState, scene2_state: SceneState):
        assert scene1_state.rgb_was_set, 'The input scene1_state must contain the RGB image'
        assert scene1_state.segmap_was_set, 'The input scene1_state must contain a segmentation map for the target object'
        assert scene1_state.depth_was_set, 'The input scene1_state must contain the depth image'
        assert scene2_state.rgb_was_set, 'The input scene2_state must contain the RGB image'
        assert scene2_state.segmap_was_set, 'The input scene2_state must contain a segmentation map for the target object'
        assert scene2_state.depth_was_set, 'The input scene2_state must contain the depth image'
