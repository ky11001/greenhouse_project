import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot


def pose_to_action(pose, gripper_opening=0., terminate=0.):
    # return np.concatenate((pose[:3, 3], Rot.from_matrix(pose[:3, :3]).as_rotvec()))
    # return np.concatenate((pose[:3, 3], Rot.from_matrix(pose[:3, :3]).as_rotvec(), [gripper_opening], [terminate]))
    return np.concatenate((pose[:3, 3], Rot.from_matrix(pose[:3, :3]).as_rotvec(), [terminate]))


def action_to_pose(action):
    pose = np.eye(4)
    pose[:3, 3] = action[:3]
    pose[:3, :3] = Rot.from_rotvec(action[3:6]).as_matrix()
    return pose


def action_scaling(actions, trans_rot_min, trans_rot_max, new_min=-1, new_max=1):
    old_range = trans_rot_max - trans_rot_min
    if isinstance(actions, torch.Tensor):
        new_range = (new_max - new_min) * torch.ones(6, dtype=actions.dtype, device=actions.device)
    else:
        new_range = (new_max - new_min) * np.ones(6)
    actions[..., :6] = (((actions[..., :6] - trans_rot_min) * new_range) / old_range) + new_min
    return actions


def action_unscaling(actions, tgt_trans_rot_min, tgt_trans_rot_max, current_min=-1, current_max=1):
    if isinstance(actions, torch.Tensor):
        old_range = (current_max - current_min) * torch.ones(6, dtype=actions.dtype, device=actions.device)
    else:
        old_range = (current_max - current_min) * np.ones(6)
    new_range = tgt_trans_rot_max - tgt_trans_rot_min
    actions[..., :6] = (((actions[..., :6] - current_min) * new_range) / old_range) + tgt_trans_rot_min
    return actions
