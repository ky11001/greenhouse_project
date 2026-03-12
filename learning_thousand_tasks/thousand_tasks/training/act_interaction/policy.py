import socket

pc_name = socket.gethostname()

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.data import Data

from thousand_tasks.training.act_interaction.config import mt_act_args, mt_act_args_dict
from thousand_tasks.training.act_interaction.action_utils import action_to_pose, pose_to_action, \
    action_unscaling, action_scaling
from thousand_tasks.models.detr_vae import build as build_ACT_model
from thousand_tasks.training.point_cloud_utils import transform_pcd_np, remove_small_clusters_from_pcd, \
    downsample_pcd
from thousand_tasks.training.py_vista_utils import draw_coord_frame


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def build_ACT_model_and_optimizer(args_override):
    args = mt_act_args

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_ACT_model(args)
    model.to(args_override['device'])

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def get_action(policy, o3d_pcd_world, T_WE, gripper_state, text_features, past_actions, plotter=None, shown=False):
    policy.eval()

    trans_rot_min = policy.trans_rot_min.cpu().numpy()
    trans_rot_max = policy.trans_rot_max.cpu().numpy()

    # Convert to numpy
    pcd_world = np.asarray(o3d_pcd_world.points)
    pcd_rgb = np.asarray(o3d_pcd_world.colors)  # This is 0 to 1

    pcd_world, pcd_rgb = remove_small_clusters_from_pcd(pcd_world, pcd_rgb, min_num_points=90)
    pcd_world, pcd_rgb = downsample_pcd(pcd_world, pcd_rgb, 2048)

    # Transform to gripper frame.
    pcd_eef = transform_pcd_np(pcd_world, np.linalg.inv(T_WE), side='left')

    # Get current state.
    curr_state = pose_to_action(T_WE, gripper_state)

    # Convert to torch tensors.
    # past_actions = np.array(
    #     [pose_to_action(pose, gripper_state) for pose, grip in zip(past_poses, past_gripper_states)])

    scaled_past_actions = past_actions.copy()
    if policy.action_mode == 'abs_delta':
        for i in range(1, policy.history_size):
            scaled_past_actions[i][:6] = pose_to_action(
                action_to_pose(scaled_past_actions[i - 1]) @ action_to_pose(scaled_past_actions[i]))[:6]

    scaled_past_actions = action_scaling(scaled_past_actions,
                                         policy.trans_rot_min_hist.cpu().numpy(),
                                         policy.trans_rot_max_hist.cpu().numpy(),
                                         )
    data = Data(
        pos=torch.tensor(pcd_eef, dtype=torch.float32),
        rgb=torch.tensor(pcd_rgb, dtype=torch.float32),
        actions=torch.Tensor([0.]),
        state=torch.tensor(curr_state, dtype=torch.float32).unsqueeze(0),
        past_actions=torch.tensor(scaled_past_actions, dtype=torch.float32).unsqueeze(0),
        is_pad=torch.Tensor([0.]),
        text_features=text_features,
        batch=torch.zeros(pcd_world.shape[0], dtype=torch.int64),
    )

    all_actions = policy(data.to('cuda')).squeeze(0).cpu().numpy()

    all_actions = action_unscaling(all_actions, trans_rot_min, trans_rot_max)

    poses = [T_WE]
    # poses_gt_past = [T_WE]

    print(f"First predicted action: {all_actions[0]}")
    for j in range(len(all_actions)):
        # processed_past_action = past_actions[-j - 1].copy()
        # poses_gt_past.append(poses_gt_past[-1] @ np.linalg.inv(action_to_pose(processed_past_action)))
        if policy.action_mode == 'delta':
            poses.append(poses[-1] @ action_to_pose(all_actions[j]))
        else:
            poses.append(poses[0] @ action_to_pose(all_actions[j]))

    waypoints = poses[1:]

    # Only change gripper state when confident
    gripper_action_prob = torch.nn.functional.sigmoid(torch.tensor(all_actions[:, -2]))

    terminate_prob = torch.nn.functional.sigmoid(torch.tensor(all_actions[:, -1]))

    if policy.action_mode == 'abs_delta':
        # Convert back to deltas. Always save in this format.
        for i in range(1, len(poses)):
            all_actions[i - 1] = pose_to_action(np.linalg.inv(poses[i - 1]) @ poses[i], all_actions[i - 1][-2],
                                            all_actions[i - 1][-1])

    past_actions = np.concatenate((past_actions[policy.open_loop_horizon:],
                                   all_actions[:policy.open_loop_horizon]), axis=0)
    past_actions[-policy.open_loop_horizon:, -2] = gripper_action_prob[:policy.open_loop_horizon] > 0.8
    past_actions[-policy.open_loop_horizon:, -1] = terminate_prob[:policy.open_loop_horizon] > 0.8

    if plotter is not None:
        plotter.add_mesh(pcd_world, color='blue', point_size=10, render_points_as_spheres=True, name='pcd')
        for k, pose in enumerate(poses):
            draw_coord_frame(plotter, pose, name=f'T_WE_{k}')
            # draw_coord_frame(plotter, poses_gt_past[k], name=f'T_WE_gt_past_{k}', opacity=0.3, scale=0.5)
            print('Fix past actions. Look at dataset get() function.')
        if not shown:
            plotter.show(auto_close=False)

    return waypoints, gripper_action_prob, terminate_prob, past_actions


def temporal_ensemble(actions_history, horizon=10, weights=None):
    if weights is None:
        weights = np.ones(horizon) / horizon

    action = np.zeros_like(actions_history[0][0])

    actions_history = np.array(actions_history, copy=True)
    actions_history[:, :, :3] /= mt_act_args_dict['trans_action_scaling']
    actions_history[:, :, 3:-1] /= mt_act_args_dict['rot_action_scaling']

    action[:3] = np.sum([actions_history[i][i][:3] * weights[i] for i in range(horizon)], axis=0)
    action[-1] = np.sum([actions_history[i][i][-1] * weights[i] for i in range(horizon)], axis=0)
    action[3:-1] = Rot.mean(Rot.from_rotvec(np.stack([actions_history[i][i][3:-1] for i in range(horizon)])),
                            weights=weights).mean().as_rotvec()

    print(f'action in f: {action}')

    return action_to_pose(action), torch.nn.functional.sigmoid(torch.tensor(action[-1]))


class ACTPolicy(nn.Module):
    def __init__(self, args_override, **kwargs):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.ignore_geoLoss = kwargs.get('ignore_geometric_loss', False)
        self.action_mode = args_override['action_mode']
        self.history_size = args_override['history_size']
        self.open_loop_horizon = args_override['open_loop_horizon']
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.bce_gripper_weight = 1.
        self.bce_terminate_weight = 1.
        self.l1_rot_weight = 1
        self.register_buffer('trans_rot_min', torch.tensor(np.ones(6), dtype=torch.float32))
        self.register_buffer('trans_rot_max', torch.tensor(np.ones(6), dtype=torch.float32))
        self.register_buffer('trans_rot_min_hist', torch.tensor(np.ones(6), dtype=torch.float32))
        self.register_buffer('trans_rot_max_hist', torch.tensor(np.ones(6), dtype=torch.float32))
        self.register_buffer('num_term_acts', torch.tensor(1., dtype=torch.float32))
        self.register_buffer('num_dont_term_acts', torch.tensor(1., dtype=torch.float32))
        self.register_buffer('num_open_acts', torch.tensor(1., dtype=torch.float32))
        self.register_buffer('num_close_acts', torch.tensor(1., dtype=torch.float32))
        print(f'KL Weight {self.kl_weight}')

        # Weights for the loss
        if self.ignore_geoLoss:
            self.var_trans = torch.tensor(1.)
            self.var_rot = torch.tensor(1.)
            self.var_gripper = torch.tensor(1.)
            self.var_termination = torch.tensor(1.)
        else:
            self.var_trans = nn.Parameter(torch.tensor(1.))
            self.var_rot = nn.Parameter(torch.tensor(1.))
            self.var_gripper = nn.Parameter(torch.tensor(1.))
            self.var_termination = nn.Parameter(torch.tensor(1.))
            self.optimizer.add_param_group(
                {'params': [self.var_trans, self.var_rot, self.var_gripper, self.var_termination]})

    def __call__(self, data):
        actions = data.actions
        is_pad = data.is_pad

        if len(actions.shape) > 1:  # training time

            a_hat, is_pad_hat, (mu, logvar) = self.model(data)

            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()

            trans_l1 = F.l1_loss(actions[..., :3], a_hat[..., :3], reduction='none')
            rot_l1 = F.l1_loss(actions[..., 3:6], a_hat[..., 3:6], reduction='none')

            if self.num_open_acts == 0:
                gripper_pos_weight = torch.tensor(1, device=self.num_open_acts.device)
            else:
                gripper_pos_weight = self.num_open_acts / self.num_close_acts

            gripper_bce = torch.nn.functional.binary_cross_entropy_with_logits(a_hat[:, :, -2],
                                                                               actions[:, :, -2],
                                                                               pos_weight=gripper_pos_weight,
                                                                               reduction='none')

            termination_bce = torch.nn.functional.binary_cross_entropy_with_logits(a_hat[:, :, -1],
                                                                                   actions[:, :, -1],
                                                                                   pos_weight=self.num_dont_term_acts / self.num_term_acts,
                                                                                   reduction='none')
            termination_bce_mean = (termination_bce * ~is_pad).mean()
            gripper_bce_mean = (gripper_bce * ~is_pad).mean()
            trans_l1_mean = (trans_l1 * ~is_pad.unsqueeze(-1)).mean()
            rot_l1_mean = (rot_l1 * ~is_pad.unsqueeze(-1)).mean()

            # Implementation of Learn weighting with homoscedastic uncertainty from
            # "Geometric loss functions for camera pose regression with deep learning"

            if self.ignore_geoLoss:
                self.var_trans = torch.tensor(1.)
                self.var_rot = torch.tensor(1.)
                self.var_gripper = torch.tensor(1.)
                self.var_termination = torch.tensor(1.)

            total_loss = (trans_l1_mean / self.var_trans
                          + rot_l1_mean / self.var_rot
                          + termination_bce_mean / self.var_termination
                          + gripper_bce_mean / self.var_gripper
                          + self.kl_weight * total_kld[0]
                          + torch.log(1 + self.var_trans)
                          + torch.log(1 + self.var_rot)
                          + torch.log(1 + self.var_termination)
                          + torch.log(1 + self.var_gripper)
                          )

            # Compute errors in mm and in deg
            with torch.no_grad():
                gt_actions = actions.clone().detach()
                pred_actions = a_hat.clone().detach()
                gt_actions = action_unscaling(gt_actions, self.trans_rot_max, self.trans_rot_min)
                pred_actions = action_unscaling(pred_actions, self.trans_rot_max, self.trans_rot_min)
                # Trans error
                trans_error = torch.norm(gt_actions[..., :3] - pred_actions[..., :3], dim=2).mean().cpu().item()
                # Rot error
                R_gt = np.array(Rot.from_rotvec(gt_actions[..., 3:6].cpu().numpy().reshape(-1, 3)).as_matrix())
                R_pred = np.array(Rot.from_rotvec(pred_actions[..., 3:6].cpu().numpy().reshape(-1, 3)).as_matrix())
                R_delta = R_gt @ R_pred.transpose(0, 2, 1)
                rotvec_delta = Rot.from_matrix(R_delta).as_rotvec()
                rot_error = np.rad2deg(np.linalg.norm(rotvec_delta, axis=1)).mean()

            loss_dict['Termination_BCE'] = termination_bce_mean
            loss_dict['Gripper_BCE'] = gripper_bce_mean
            loss_dict['Trans_L1'] = trans_l1_mean
            loss_dict['Rot_L1'] = rot_l1_mean
            loss_dict['KL'] = total_kld[0]
            loss_dict['loss'] = total_loss
            loss_dict['Rot_Error'] = rot_error
            loss_dict['Trans_Error'] = trans_error
            return loss_dict

        else:  # inference time
            a_hat, _, (_, _) = self.model(data)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer
