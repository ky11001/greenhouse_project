import argparse
import os
import socket
import sys
from os.path import join

import clip
import numpy as np
import torch
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Data
from torch_geometric.nn import fps, nearest

sys.path.append('/home/kamil/phd/osil/')  # required on tiger

from thousand_tasks.core.globals import ASSETS_DIR
from thousand_tasks.core.utils.se3_tools import posevec2pose, rotvec2rot, rot2rotvec
from thousand_tasks.training.point_cloud_utils import transform_pcd_np, \
    backproject_camera_target_realworld, remove_small_clusters_from_pcd, downsample_pcd
from thousand_tasks.training.act_interaction.action_utils import action_to_pose, pose_to_action, \
    action_unscaling, action_scaling
from thousand_tasks.data.utils import remove_demo_number_from_task_folder_name, get_skill_name


class Evaluator:

    def __init__(self, root, config, which_camera: str = 'head'):
        # Dataset root and task directories
        self.root = root
        self.task_dirs = [d for d in os.listdir(root) if os.path.isdir(
            join(root, d)) and 'processed' not in d]
        self.T_WC_external = np.load(join(ASSETS_DIR, f'T_WC_{which_camera}.npy'))

        self.done_thresh = 0.8
        self.open_gripper_thresh = 0.2
        self.close_gripper_thresh = 0.8

        self.min_num_points_per_cluster = 300
        self.not_terminate_action = 0.
        self.open_gripper_action = 0.
        self.closed_gripper_action = 1.
        self.open_loop_horizon = None

        self._policy = None
        self.trans_rot_min_torch = None
        self.trans_rot_max_torch = None
        self.trans_rot_min_np = None
        self.trans_rot_max_np = None

        self.trans_rot_min_hist_torch = None
        self.trans_rot_max_hist_torch = None
        self.trans_rot_min_hist_np = None
        self.trans_rot_max_hist_np = None

        self.action_mode = config['action_mode']
        self.history_size = config['history_size']
        self.max_num_points = config['max_num_points']
        self.trans_perturb_to_bn_lb = config['eval_trans_perturb_to_bn_lb']
        self.trans_perturb_to_bn_ub = config['eval_trans_perturb_to_bn_ub']
        self.rot_perturb_to_bn_lb = config['eval_rot_perturb_to_bn_lb']
        self.rot_perturb_to_bn_ub = config['eval_rot_perturb_to_bn_ub']
        self.pcd_noise_std = config['preprocessing_pcd_noise_std']
        self.total_num_clusters = config['preprocessing_num_clusters_total']
        self.num_clusters_to_mask = config['preprocessing_num_clusters_to_mask']

        # # Neutral pose
        # self.neutral_pose = np.array(
        #     [[0.0015094184834770763, -0.9998804337478877, -0.015389602463277929, 0.5742849099508102],
        #      [-0.9999968390101373, -0.0015401833411734511, 0.0019874116356231536, -0.0497090906055144],
        #      [-0.002010876817603699, 0.015386553981043462, -0.9998795979171757, 0.49416191861279085],
        #      [0.0, 0.0, 0.0, 1.0]])

        # Trajectory initialisation
        self.current_task_name = None
        self.current_task_idx = None
        self.eef_pose = None
        self.action_history = None
        self.done = None
        self.traj = None

        # Clip
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device='cpu')

        self.task_names = []

        # Load point clouds and bottleneck poses
        self.gt_traj = {}
        self.pcds_world = {}
        self.pcds_rgb = {}
        self.bottleneck_poses = {}
        self.text_features = {}

        for task_folder_name in self.task_dirs:
            task_dir = join(self.root, task_folder_name)

            task_name = remove_demo_number_from_task_folder_name(task_folder_name)
            skill_name = get_skill_name(task_dir)

            if task_name not in self.task_names:
                self.task_names.append(task_name)

                text_tokens = clip.tokenize([skill_name.replace('_', ' ')])
                with torch.no_grad():
                    self.text_features[task_name] = self.clip_model.encode_text(text_tokens).float()

                self.pcds_world[task_name] = []
                self.pcds_rgb[task_name] = []
                self.bottleneck_poses[task_name] = []
                self.gt_traj[task_name] = []

            self.bottleneck_poses[task_name].append(
                posevec2pose(np.load(join(task_dir, 'demo_eef_posevecs.npy'))[0][:-1]))

            rgb_init = np.array(Image.open(join(task_dir, f'{which_camera}_camera_ws_rgb.png'))) / 255
            depth_init = np.array(Image.open(join(task_dir, f'{which_camera}_camera_ws_depth_to_rgb.png')))
            mask_init = np.load(join(task_dir, f'{which_camera}_camera_ws_segmap.npy'))
            external_camera_intrinsics = np.load(join(task_dir, f'{which_camera}_camera_rgb_intrinsic_matrix.npy'))

            pcd_cam_initial, pcd_rgb_initial = backproject_camera_target_realworld(
                im_depth=depth_init / 1000,
                rgb=rgb_init,
                K=external_camera_intrinsics,
                target_mask=np.logical_not(mask_init))
            pcd_cam, pcd_rgb = remove_small_clusters_from_pcd(pcd_cam_initial,
                                                              pcd_rgb_initial,
                                                              self.min_num_points_per_cluster)

            pcd_world = transform_pcd_np(pcd_cam, self.T_WC_external, side='left')

            self.pcds_world[task_name].append(torch.tensor(pcd_world, dtype=torch.float32))
            self.pcds_rgb[task_name].append(torch.tensor(pcd_rgb, dtype=torch.float32))
            self.gt_traj[task_name].append(np.load(join(task_dir, 'sampled_eef_poses.npy')))

    @property
    def policy(self):
        return self._policy

    @policy.setter
    def policy(self, policy):
        self._policy = policy
        self.open_loop_horizon = self._policy.open_loop_horizon
        self.trans_rot_min_torch = policy.trans_rot_min.clone()
        self.trans_rot_max_torch = policy.trans_rot_max.clone()
        self.trans_rot_min_np = self.trans_rot_min_torch.cpu().numpy()
        self.trans_rot_max_np = self.trans_rot_max_torch.cpu().numpy()

        self.trans_rot_min_hist_torch = policy.trans_rot_min_hist.clone()
        self.trans_rot_max_hist_torch = policy.trans_rot_max_hist.clone()
        self.trans_rot_min_hist_np = self.trans_rot_min_hist_torch.cpu().numpy()
        self.trans_rot_max_hist_np = self.trans_rot_max_hist_torch.cpu().numpy()

    def init_trajectory(self, task_name):
        # self.shown = False
        self.done = False
        self.current_task_name = task_name
        self.current_task_idx = np.random.randint(len(self.pcds_world[task_name]))
        self.action_history = np.array(
            [[0., 0., 0., 0., 0., 0., self.closed_gripper_action, self.not_terminate_action]] * self.history_size)

        self.eef_pose = self.bottleneck_poses[task_name][self.current_task_idx].copy()
        self.eef_pose[:3, 3] += np.random.uniform(self.trans_perturb_to_bn_lb, self.trans_perturb_to_bn_ub)
        self.eef_pose[:3, :3] = rotvec2rot(
            np.random.uniform(self.rot_perturb_to_bn_lb, self.rot_perturb_to_bn_ub)) @ self.eef_pose[:3, :3]

        self.traj = [self.eef_pose.copy()]

    def step(self, policy):
        # Current pose
        T_WE = self.eef_pose

        # Point cloud
        pcd_pos = self.pcds_world[self.current_task_name][self.current_task_idx].clone()
        pcd_rgb = self.pcds_rgb[self.current_task_name][self.current_task_idx].clone()

        # Mask out portions of the point cloud
        indices = fps(pcd_pos, ratio=self.total_num_clusters / len(pcd_pos))
        cluster_idx = nearest(pcd_pos, pcd_pos[indices])

        num_clusters_to_mask = np.random.randint(1, self.num_clusters_to_mask + 1)
        cluster_indices_to_keep = np.random.choice([cls_idx for cls_idx in range(self.total_num_clusters)],
                                                   size=self.total_num_clusters - num_clusters_to_mask,
                                                   replace=False)
        keep = np.isin(cluster_idx, cluster_indices_to_keep)
        pcd_pos, pcd_rgb = pcd_pos[keep], pcd_rgb[keep]

        # Down sample
        pcd_pos, pcd_rgb = downsample_pcd(pcd_pos, pcd_rgb, self.max_num_points)

        # Add gaussian noise to point cloud
        pcd_pos = pcd_pos + torch.randn_like(pcd_pos) * self.pcd_noise_std

        # Convert to numpy
        pcd_world = np.asarray(pcd_pos)
        pcd_rgb = np.asarray(pcd_rgb)
        # self.plotter.add_mesh(pv.PolyData(pcd_world), color='red', render_points_as_spheres=True, point_size=5,
        #                       name='pcd')
        # Transform to gripper frame.
        pcd_eef = transform_pcd_np(pcd_world, np.linalg.inv(T_WE), side='left')

        # Get current state.
        curr_state = pose_to_action(T_WE, 1, self.done)

        if self.action_mode == 'abs_delta':
            past_actions = self.action_history.copy()
            for i in range(1, self.history_size):
                past_actions[i][:6] = pose_to_action(
                    action_to_pose(past_actions[i - 1]) @ action_to_pose(past_actions[i]))[:6]
        else:
            past_actions = self.action_history.copy()

        scaled_past_actions = action_scaling(past_actions, self.trans_rot_min_hist_np, self.trans_rot_max_hist_np)

        data = Data(
            pos=torch.tensor(pcd_eef, dtype=torch.float32),
            rgb=torch.tensor(pcd_rgb, dtype=torch.float32),
            actions=torch.Tensor([0.]),
            state=torch.tensor(curr_state, dtype=torch.float32).unsqueeze(0),
            past_actions=torch.tensor(scaled_past_actions, dtype=torch.float32).unsqueeze(0),
            is_pad=torch.Tensor([0.]),
            text_features=self.text_features[self.current_task_name],
            batch=torch.zeros(pcd_world.shape[0], dtype=torch.int64),
        )

        with torch.no_grad():
            if isinstance(policy, DDP):
                ddp_device = policy.module.model.action_head.weight.device
                scaled_pred_actions = policy(data.to(ddp_device)).squeeze(0).cpu().numpy()
            else:
                scaled_pred_actions = policy(data.to(policy.model.action_head.weight.device)).squeeze(0).cpu().numpy()

        pred_actions = action_unscaling(scaled_pred_actions, self.trans_rot_min_np, self.trans_rot_max_np)

        poses = [T_WE]
        for j in range(len(pred_actions)):
            if self.action_mode == 'abs_delta':
                poses.append(poses[0] @ action_to_pose(pred_actions[j]))
            else:
                poses.append(poses[-1] @ action_to_pose(pred_actions[j]))

        waypoints = poses[1:]

        # for k, pose in enumerate(poses):
        #     draw_coord_frame(self.plotter, pose, name=f'{k}')
        #
        # if not self.shown:
        #     self.plotter.show(auto_close=False)
        #     self.shown = True

        gripper_prob = torch.nn.functional.sigmoid(torch.tensor(pred_actions[:, -2]))
        terminate_prob = torch.nn.functional.sigmoid(torch.tensor(pred_actions[:, -1]))

        if self.action_mode == 'abs_delta':
            # Convert back to deltas. Always save in this format.
            for i in range(1, len(poses)):
                pred_actions[i - 1] = pose_to_action(np.linalg.inv(poses[i - 1]) @ poses[i], pred_actions[i - 1, -2],
                                                     pred_actions[i - 1, -1])

        self.action_history = np.concatenate((self.action_history[self.open_loop_horizon:],
                                              pred_actions[:self.open_loop_horizon]), axis=0)

        self.action_history[-self.open_loop_horizon:, -1] = terminate_prob[:self.open_loop_horizon] > self.done_thresh

        for offset, prob in enumerate(gripper_prob[:self.open_loop_horizon]):
            if prob < self.open_gripper_thresh:
                self.action_history[-self.open_loop_horizon + offset, -2] = self.open_gripper_action
            elif prob > self.close_gripper_thresh:
                self.action_history[-self.open_loop_horizon + offset, -2] = self.closed_gripper_action
            else:
                self.action_history[-self.open_loop_horizon + offset, -2] = self.action_history[-self.open_loop_horizon + offset - 1, -2]

        for i in range(self.open_loop_horizon):
            self.eef_pose = waypoints[i]  # Update eef pose
            self.traj.append(self.eef_pose.copy())

            if terminate_prob[i] > self.done_thresh:
                self.done = True
                break

    def evaluate_policy(self, policy, num_rollouts_per_task=10, return_individual_results=False):

        if isinstance(policy, DDP):
            self.policy = policy.module
        else:
            self.policy = policy

        pos_errors = {}  # In mm
        rot_errors = {}  # In deg

        # Loop over all tasks
        for task_name in self.task_names:

            # draw_coord_frame(self.plotter, self.bottleneck_poses[task_dir], name='bottleneck_pose', scale=2)

            pos_errors[task_name] = []
            rot_errors[task_name] = []

            # Loop over different trajectories
            for _ in range(num_rollouts_per_task):

                # Initialise a trajectory
                self.init_trajectory(task_name)

                counter = 0

                while self.done is False:
                    self.step(policy)
                    counter += 1
                    if counter > 100:
                        self.done = True

                trans_error, rot_error = get_trajectory_error(np.array(self.traj), self.gt_traj[self.current_task_name][
                    self.current_task_idx])

                pos_errors[task_name].append(trans_error * 1e3)
                rot_errors[task_name].append(rot_error)

                # self.plotter.show(auto_close=False)

            pos_errors[task_name] = np.mean(pos_errors[task_name])
            rot_errors[task_name] = np.mean(rot_errors[task_name])

        if return_individual_results:
            return pos_errors, rot_errors
        else:
            return np.mean(list(pos_errors.values())), np.mean(list(rot_errors.values()))


def get_single_trajectory_error(traj_1: np.ndarray, traj_2: np.ndarray):
    np_wps_1 = len(traj_1)

    # Get distance between each waypoint in trajectory 1 and each waypoint in trajectory 2
    dists = np.linalg.norm(traj_1[:, None, :3, 3] - traj_2[None, :, :3, 3], axis=2)

    # Get index of closest point in trajectory 2 for each point in trajectory 1
    indices_2 = np.argmin(dists, axis=1)

    # Compute sum of translation errors
    trans_error = dists[np.arange(np_wps_1), indices_2].sum()

    # Compute rotation error
    rot_error = 0  # deg
    for idx in range(np_wps_1):
        R_delta = traj_1[idx, :3, :3] @ traj_2[indices_2[idx], :3, :3].T
        rot_error += np.rad2deg(np.linalg.norm(rot2rotvec(R_delta)))

    return trans_error / np_wps_1, rot_error / np_wps_1


def get_trajectory_error(traj_1: np.ndarray, traj_2: np.ndarray):
    trans_error_1, rot_error_1 = get_single_trajectory_error(traj_1, traj_2)
    trans_error_2, rot_error_2 = get_single_trajectory_error(traj_2, traj_1)
    return (trans_error_1 + trans_error_2) / 2, (rot_error_1 + rot_error_2) / 2


if __name__ == '__main__':
    # --- Arg parser ---
    parser = argparse.ArgumentParser(description="MT-ACT training script")
    parser.add_argument('-name', '--name', type=str, required=False, help='policy_name')
    parser.add_argument('-dir', '--dir', type=str, required=False, help='dataset dir name')
    parser.add_argument('-device', '--device', type=str, required=False, help='Device name: cpu, cuda or mps for mac',
                        default='cuda')
    args = parser.parse_args()
    # --- End ---

    ############################################################################################################
    pc_name = socket.gethostname()

    if pc_name == 'slifer':
        dset_root = join('/home/kamil/phd/head_cam/assets', args.dir)
        policy_path = f'/home/kamil/phd/head_cam/thousand_tasks/baselines/mt_act/interaction/runs/{args.name}/{args.name}.pt'
    elif pc_name == 'tiger' or pc_name == 'omen':
        dset_root = join('/home/kamil/phd/osil/assets', args.dir)
        policy_path = f'/baselines/interaction/runs/{args.name}/{args.name}.pt'
    else:
        dset_root = join('/mnt/data/kamil', args.dir)
        policy_path = f'/baselines/interaction/runs/{args.name}/{args.name}.pt'

    # mt_act_args_dict['device'] = args.device
    # policy = ACTPolicy(mt_act_args_dict)
    # policy.to(args.device)
    #
    # # load weights
    # policy.load_state_dict(torch.load(policy_path))
    #
    # generator = Evaluator(dset_root, mt_act_args_dict, policy)
