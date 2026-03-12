import socket

pc_name = socket.gethostname()

import os
from os.path import join
import pyvista as pv
import pickle

import clip
import numpy as np
import torch
from PIL import Image
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import fps, nearest
from tqdm import tqdm

from thousand_tasks.core.utils.se3_tools import posevec2pose, rotvec2rot
from thousand_tasks.core.globals import ASSETS_DIR
from thousand_tasks.training.act_bn_reaching.action_utils import (pose_to_action,
                                                                   action_to_pose,
                                                                   action_scaling,
                                                                   action_unscaling)
from thousand_tasks.training.point_cloud_utils import transform_pcd_np, \
    backproject_camera_target_realworld, remove_small_clusters_from_pcd, downsample_pcd
from thousand_tasks.training.py_vista_utils import draw_coord_frame
from thousand_tasks.data.utils import printarr, interpolate_poses, get_skill_name

def opener(path, flags):
    return os.open(path, flags, 0o770)


def parse_dataset_statistics(path):
    f = open(path, "r")
    ds_len = int(f.readline().split(': ')[-1].split('\n')[0])
    num_bn_pts = int(f.readline().split(': ')[-1].split('\n')[0])
    num_demo_pts = int(f.readline().split(': ')[-1].split('\n')[0])
    f.readline()
    f.readline()
    num_term_acts = int(f.readline().split(': ')[-1].split('\n')[0])
    num_not_term_acts = int(f.readline().split(': ')[-1].split('\n')[0])
    f.readline()
    num_open_acts = int(f.readline().split(': ')[-1].split('\n')[0])
    num_close_acts = int(f.readline().split(': ')[-1].split('\n')[0])

    return ds_len, num_bn_pts, num_demo_pts, num_term_acts, num_not_term_acts, num_open_acts, num_close_acts


def get_mean_distance_between_waypoints_in_demon(traj_eef_posevec):
    deltas = traj_eef_posevec[:-1, :3] - traj_eef_posevec[1:, :3]
    dist_between_waypoints = np.linalg.norm(deltas, axis=1)
    print(
        f'Mean distance between waypoints: ({np.mean(dist_between_waypoints) * 1e3:.3f} +- {np.std(dist_between_waypoints) * 1e3:.3f}) mm')


def get_mean_distance_between_waypoints_in_all_demons(tasks_dir_path):
    task_names = [d for d in os.listdir(tasks_dir_path) if os.path.isdir(
        join(tasks_dir_path, d)) and d != 'bn_reaching_processed' and d != 'interaction_processed' and d != 'processed']

    dists = []

    for task_name in task_names:
        task_dir_path = join(tasks_dir_path, task_name)
        traj_eef_posevec = np.load(join(task_dir_path, 'demo_eef_posevecs.npy'))

        deltas = traj_eef_posevec[:-1, :3] - traj_eef_posevec[1:, :3]
        dist_between_waypoints = np.linalg.norm(deltas, axis=1)

        dists += dist_between_waypoints.tolist()

    print(f'Mean distance between waypoints: ({np.mean(dists) * 1e3:.3f} +- {np.std(dists) * 1e3:.3f}) mm')

    return np.mean(dists), np.std(dists)


def get_subsample_indices(traj_eef_posevec, target_dist=0.01):
    # Figure out precise indices at which the gripper state changes. For each idx, gripper state was changed between
    # traj_eef_posevec[idx - 1] and traj_eef_posevec[idx]
    gripper_states = traj_eef_posevec[:, -1]
    delta_gripper_states = np.abs(gripper_states[1:] - gripper_states[:-1])
    delta_gripper_indices = np.flatnonzero(delta_gripper_states)
    delta_gripper_indices += 1

    # Get the distance between all waypoints
    dists_between_waypoints = np.linalg.norm(traj_eef_posevec[1:, :3] - traj_eef_posevec[:-1, :3], axis=1)

    waypoint_indices = [0]

    while waypoint_indices[-1] != len(traj_eef_posevec) - 1:
        # Get cumulative distances
        if waypoint_indices[-1] == 0:
            cumsum_dists = np.cumsum(dists_between_waypoints)
        else:
            cumsum_dists = np.cumsum(dists_between_waypoints[waypoint_indices[-1]:])

        # Calculate delta to desired distance
        delta_target_dist = cumsum_dists - target_dist

        # If all delta distances are negative, the final waypoint is closer than the desired distance
        if (delta_target_dist < 0).all():
            waypoint_idx = len(traj_eef_posevec) - 1
            # idx = len(delta_target_dist) - 1

        # If all delta distances are positive, first waypoint is further than the desired distance
        elif (delta_target_dist > 0).all():
            waypoint_idx = waypoint_indices[-1] + 1

        else:
            # See when delta changes from positive to negative.
            # see https://stackoverflow.com/questions/61233411/find-indices-where-a-python-array-becomes-positive-but-not-negative
            m1 = delta_target_dist[:-1] < 0
            m2 = np.sign(delta_target_dist[1:] * delta_target_dist[:-1]) == -1

            r = np.argwhere(np.all(np.vstack([m1, m2]), axis=0))
            idx = np.squeeze(r).tolist() + 1
            waypoint_idx = idx + waypoint_indices[-1] + 1
        # print()
        if len(delta_gripper_indices) > 0 and waypoint_idx > delta_gripper_indices[0]:
            waypoint_idx = delta_gripper_indices[0]
            delta_gripper_indices = np.delete(delta_gripper_indices, 0)
            # print('Gripper state changed')
        waypoint_indices.append(waypoint_idx)

        # print(f'Moved distance    : {cumsum_dists[idx] * 1e3:.2f} (target distance {target_dist * 1e3:.2f})')
        # eucledian_dist = np.linalg.norm(
        #     traj_eef_posevec[waypoint_indices[-2], :3] - traj_eef_posevec[waypoint_indices[-1], :3])
        # print(f'Euclidean distance: {eucledian_dist * 1e3:.2f} (target distance {target_dist * 1e3:.2f})')

    # Check if last two and first two waypoints can be merged
    traj_eef_posevec_sampled = traj_eef_posevec[waypoint_indices]
    dists_between_sampled_waypoints = np.linalg.norm(
        traj_eef_posevec_sampled[1:, :3] - traj_eef_posevec_sampled[:-1, :3], axis=1)

    # Check first and second waypoint
    if dists_between_sampled_waypoints[:2].sum() < target_dist * 1.3:
        # print('Merging first two waypoints')
        waypoint_indices = waypoint_indices[0:1] + waypoint_indices[2:]  # remove second index

    # Check last two elements
    if dists_between_sampled_waypoints[-2:].sum() < target_dist * 1.3:
        # print('Merging last two waypoints')
        waypoint_indices = waypoint_indices[:-2] + waypoint_indices[-1:]

    return waypoint_indices


class BCDataset(Dataset):
    def __init__(self, root, config, num_traj_to_gen=None, transform=None, pre_transform=None, reprocess=False,
                 mask_pcd_clusters=True, add_noise_to_pcd=True, add_noise_to_act_hist=True,
                 processed_dir='processed', visualise=False, which_camera: str = 'external',
                 dp_per_chunk: int = 500):

        self.dp_per_chunk = dp_per_chunk
        dp_per_trj = 78.5  # empirical average
        self.chunk = []
        self.loaded_chunks = None
        self._batch_size = None

        self.action_mode = config['action_mode']
        # Either T_{t, t+1} or T_{0, t+1}
        assert self.action_mode in ['delta', 'abs_delta']

        self.mask_pcd_clusters = mask_pcd_clusters
        self.add_noise_to_pcd = add_noise_to_pcd
        self.add_noise_to_act_hist = add_noise_to_act_hist
        self.act_hist_noise_std = config['preprocessing_act_hist_noise_std']
        self.num_traj_to_bn = num_traj_to_gen

        # Not used anymore
        # self.num_extra_samples_per_demo = config['num_extra_samples_per_demo']
        self.which_camera = which_camera

        self.root = root
        self.task_dirs = [d for d in os.listdir(root) if os.path.isdir(
            join(root, d)) and 'processed' not in d]

        self.reprocess = reprocess
        self.processed_ = processed_dir

        self.chunk_size = config['chunk_size']
        self.history_size = config['history_size']

        self.T_WC = np.load(join(ASSETS_DIR, f'T_WC_{self.which_camera}.npy'))

        if self.reprocess:
            processing_dir = join(root, self.processed_)
            if os.path.isdir(processing_dir):
                data_points = [int(name[5:-4]) for name in os.listdir(processing_dir)
                               if name[-4:] == '.pkl']
                self.dataset_length = (max(data_points) + 1) if data_points else 0
            else:
                self.dataset_length = 0
            dp_to_gen = num_traj_to_gen * len(self.task_dirs) * dp_per_trj
            self.total_chunks_to_gen = np.ceil(
                dp_to_gen / self.dp_per_chunk)
            # self.total_chunks_to_gen = 8
        else:
            # -9 for pre_filter, pre_transform, trans_and_rot std, mean, min, max, dset statistics
            self._n_chunks = len(os.listdir(join(root, self.processed_))) - 10
            self.dataset_length = self._n_chunks * self.dp_per_chunk

        self.total_num_clusters = config['preprocessing_num_clusters_total']
        self.num_clusters_to_mask = config['preprocessing_num_clusters_to_mask']
        self.pcd_noise_std = config['preprocessing_pcd_noise_std']

        self.max_num_points = config['max_num_points']
        self.num_points_to_save = config['num_points_to_save']

        self.visualise = visualise
        if self.visualise:
            self.plotter = pv.Plotter()
            self.shown = False

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device='cpu')

        try:
            self.mean_counter = int(torch.load(join(self.processed_dir, 'mean_counter.pt')))
            self.trans_rot_mean = torch.load(join(self.processed_dir, 'trans_and_rot_mean.pt')).to(torch.float32)
            self.trans_rot_std = torch.load(join(self.processed_dir, 'trans_and_rot_std.pt')).to(torch.float32)
            self.trans_rot_min = torch.load(join(self.processed_dir, 'trans_and_rot_min.pt')).to(torch.float32)
            self.trans_rot_max = torch.load(join(self.processed_dir, 'trans_and_rot_max.pt')).to(torch.float32)

            self.trans_rot_min_hist = torch.load(join(self.processed_dir, 'trans_and_rot_min_hist.pt')).to(
                torch.float32)
            self.trans_rot_max_hist = torch.load(join(self.processed_dir, 'trans_and_rot_max_hist.pt')).to(
                torch.float32)

            _, self.num_bn_pts, self.num_demo_pts, self.num_term_acts, self.num_dont_term_acts, \
                self.num_open_acts, self.num_close_acts = parse_dataset_statistics(
                join(self.processed_dir, 'statistics.txt'))
        except OSError as _:
            self.mean_counter = 0
            self.trans_rot_mean = torch.zeros(6)
            self.trans_rot_std = torch.zeros(6)
            self.trans_rot_min = torch.inf * torch.ones(6)
            self.trans_rot_max = - torch.inf * torch.ones(6)
            self.trans_rot_min_hist = torch.inf * torch.ones(6)
            self.trans_rot_max_hist = - torch.inf * torch.ones(6)

            self.num_bn_pts, self.num_demo_pts, self.num_term_acts, self.num_dont_term_acts, \
                self.num_open_acts, self.num_close_acts = 0, 0, 0, 0, 0, 0

        self.trans_perturb_to_bn_lb = config['trans_perturb_to_bn_lb']
        self.trans_perturb_to_bn_ub = config['trans_perturb_to_bn_ub']
        self.rot_perturb_to_bn_lb = config['rot_perturb_to_bn_lb']
        self.rot_perturb_to_bn_ub = config['rot_perturb_to_bn_ub']

        self.trans_spacing_between_waypoints = config['trans_spacing_between_waypoints']
        self.corrective_action_ratio = config['corrective_action_ratio']

        self.neutral_pose = np.array(
            [[0.0015094184834770763, -0.9998804337478877, -0.015389602463277929, 0.5742849099508102],
             [-0.9999968390101373, -0.0015401833411734511, 0.0019874116356231536, -0.0497090906055144],
             [-0.002010876817603699, 0.015386553981043462, -0.9998795979171757, 0.49416191861279085],
             [0.0, 0.0, 0.0, 1.0]])

        self.min_num_points_per_cluster = 300
        self.open_gripper_action = 0.
        self.closed_gripper_action = 1.
        self.terminate_action = 1.
        self.not_terminate_action = 0.

        super(BCDataset, self).__init__(root, transform, pre_transform)

    @property
    def n_chunks(self):
        return self._n_chunks

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, bs):
        self._batch_size = bs

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        return self.raw_names

    @property
    def processed_file_names(self):
        if self.reprocess:
            return [' ']
        return [f'data_{i}.pkl' for i in range(self.n_chunks)]

    @property
    def processed_dir(self) -> str:
        return join(self.root, self.processed_)

    def load_demo_data(self, task_dir,
                       return_task_name: bool = True,
                       return_rgb_init: bool = True,
                       return_depth_init: bool = True,
                       return_mask_init: bool = True,
                       return_traj_rgb: bool = True,
                       return_traj_depth: bool = True,
                       return_traj_masks: bool = True,
                       return_traj_eef_posevec: bool = True,
                       return_traj_eef_poses: bool = True,
                       return_traj_low_dim_eff_actions: bool = True,
                       return_bottleneck_idx: bool = True,
                       return_external_camera_intrinsics: bool = True,
                       return_bottleneck_pose: bool = True):

        task_dir = join(self.root, task_dir)
        task_name = get_skill_name(task_dir).lower() if return_task_name else None

        rgb_init = np.array(Image.open(join(task_dir, f'{self.which_camera}_camera_ws_rgb.png'))) \
            if return_rgb_init else None
        depth_init = np.array(Image.open(join(task_dir, f'{self.which_camera}_camera_ws_depth_to_rgb.png'))) \
            if return_depth_init else None
        mask_init = np.load(join(task_dir, f'{self.which_camera}_camera_ws_segmap.npy')) \
            if return_mask_init else None

        traj_eef_posevec = None
        if (return_traj_eef_posevec or return_traj_rgb or
                return_traj_depth or return_traj_masks or
                return_traj_eef_poses or return_traj_low_dim_eff_actions):

            traj_eef_posevec = np.load(join(task_dir, 'demo_eef_posevecs.npy'))
            subsample_indices = get_subsample_indices(traj_eef_posevec, self.trans_spacing_between_waypoints)

            # get_mean_distance_between_waypoints_in_demon(traj_eef_posevec)
            traj_eef_posevec = traj_eef_posevec[subsample_indices]
            # get_mean_distance_between_waypoints_in_demon(traj_eef_posevec)

            traj_rgb = np.load(join(task_dir, f'{self.which_camera}_camera_rgb.npy'))[subsample_indices] \
                if return_traj_rgb else None
            traj_depth = np.load(join(task_dir, f'{self.which_camera}_camera_depth_to_rgb.npy'))[subsample_indices] \
                if return_traj_rgb else None
            traj_masks = np.load(join(task_dir, f'{self.which_camera}_camera_masks.npy'))[subsample_indices] \
                if return_traj_rgb else None

            traj_eef_poses = None
            if (return_traj_eef_poses or
                    return_traj_low_dim_eff_actions or
                    return_bottleneck_pose):
                traj_eef_poses = np.array([posevec2pose(p[:-1]) for p in traj_eef_posevec])

            bottleneck_idx = 0

            # TODO change to Tip frame here
            # Convert T_{E_t E_t+1} to actions.
            traj_low_dim_eff_actions = None
            if return_traj_low_dim_eff_actions:
                traj_low_dim_eff_actions = [pose_to_action(
                    pose=np.linalg.inv(traj_eef_poses[i]) @ traj_eef_poses[i + 1],
                    gripper_opening=traj_eef_posevec[i + 1][-1],
                    terminate=self.not_terminate_action) for i in range(len(traj_eef_posevec) - 1)]

                # Last action is do nothing, leave gripper state unchanged and terminate = 1
                traj_low_dim_eff_actions.append(pose_to_action(pose=np.eye(4),
                                                               gripper_opening=traj_eef_posevec[-1][-1],
                                                               terminate=self.terminate_action))
                traj_low_dim_eff_actions = np.array(traj_low_dim_eff_actions)

        external_camera_intrinsics = None
        if return_external_camera_intrinsics:
            external_camera_intrinsics = np.load(
                join(self.root, task_dir, f'{self.which_camera}_camera_rgb_intrinsic_matrix.npy'))
        # bottleneck_pose = np.load(join(self.root, task_dir, 'bottleneck_pose.npy'))
        bottleneck_pose = traj_eef_poses[0] if return_bottleneck_pose else None

        # Normalize rbg to [0, 1], now it is [0, 255].
        rgb_init = rgb_init / 255 if return_rgb_init else None
        traj_rgb = traj_rgb / 255. if return_traj_rgb else None

        return (
            task_name, rgb_init, depth_init, mask_init, traj_rgb, traj_depth, traj_masks, traj_eef_posevec,
            traj_eef_poses, traj_low_dim_eff_actions, bottleneck_idx, external_camera_intrinsics, bottleneck_pose)

    # def save_data_point(self, idx, pcd_eef, pcd_rgb, low_dim_action_labels, past_actions, current_pose, is_pad,
    #                     text_features, interaction_trajectory):
    #
    #     if self.action_mode == 'abs_delta':
    #         # Converting T_{t, t+1} to T_{0, t+1}
    #         all_T_e_e_1 = [action_to_pose(action) for action in low_dim_action_labels]
    #         for i in range(1, len(all_T_e_e_1)):
    #             all_T_e_e_1[i] = all_T_e_e_1[i - 1] @ all_T_e_e_1[i]
    #         low_dim_action_labels = [pose_to_action(pose, gripper_opening, terminate) for
    #                                  pose, gripper_opening, terminate in
    #                                  zip(all_T_e_e_1, low_dim_action_labels[:, -2], low_dim_action_labels[:, -1])]
    #         # Doing the same for the past actions
    #         # Now past actions are T{t-h, t-h+1}, T{t-h+1, t-h+2}... T{t-1, t}
    #         # They will be inputted as T_{t-h, t-h+1}, T_{t-h, t-h+2}... T_{t-h, t}
    #         all_past_T_e_e_1 = [action_to_pose(action) for action in past_actions]
    #         for i in range(1, len(all_past_T_e_e_1)):
    #             all_past_T_e_e_1[i] = all_past_T_e_e_1[i - 1] @ all_past_T_e_e_1[i]
    #
    #         past_actions = [pose_to_action(pose, gripper_opening, terminate) for pose, gripper_opening, terminate in
    #                         zip(all_past_T_e_e_1, past_actions[:, -2], past_actions[:, -1])]
    #     low_dim_action_labels = np.array(low_dim_action_labels)
    #     past_actions = np.array(past_actions)
    #
    #     self.trans_rot_min = torch.minimum(self.trans_rot_min,
    #                                        torch.tensor(low_dim_action_labels[:, :6].min(axis=0)))
    #     self.trans_rot_max = torch.maximum(self.trans_rot_max,
    #                                        torch.tensor(low_dim_action_labels[:, :6].max(axis=0)))
    #
    #     self.trans_rot_min_hist = torch.minimum(self.trans_rot_min_hist,
    #                                             torch.tensor(past_actions[:, :6].min(axis=0)))
    #     self.trans_rot_max_hist = torch.maximum(self.trans_rot_max_hist,
    #                                             torch.tensor(past_actions[:, :6].max(axis=0)))
    #
    #     data = Data(
    #         pos=torch.tensor(pcd_eef, dtype=torch.float32),
    #         rgb=torch.tensor(pcd_rgb, dtype=torch.float32),
    #         actions=torch.tensor(np.array(low_dim_action_labels), dtype=torch.float32).unsqueeze(0),
    #         past_actions=torch.tensor(past_actions, dtype=torch.float32).unsqueeze(0),
    #         current_pose=torch.tensor(current_pose, dtype=torch.float32).unsqueeze(0),
    #         is_pad=torch.tensor(is_pad, dtype=torch.bool).unsqueeze(0),  # Not used
    #         text_features=text_features,
    #         interaction_trajectory=torch.tensor([interaction_trajectory]).unsqueeze(0)
    #     )
    #
    #     torch.save(data, join(self.processed_dir, 'data_{}.pt'.format(idx)))

    def save_data_point(self, pcd_eef, pcd_rgb, low_dim_action_labels, past_actions, current_pose, is_pad,
                        text_features, interaction_trajectory):

        updated_dset_len = self.dataset_length
        while os.path.isfile('data_{}.pkl'.format(updated_dset_len)):
            updated_dset_len += 1

        if self.action_mode == 'abs_delta':
            # Converting T_{t, t+1} to T_{0, t+1}
            all_T_e_e_1 = [action_to_pose(action) for action in low_dim_action_labels]
            for i in range(1, len(all_T_e_e_1)):
                all_T_e_e_1[i] = all_T_e_e_1[i - 1] @ all_T_e_e_1[i]
            low_dim_action_labels = [pose_to_action(pose, gripper_opening, terminate) for
                                     pose, gripper_opening, terminate in
                                     zip(all_T_e_e_1, low_dim_action_labels[:, -2], low_dim_action_labels[:, -1])]
            # Doing the same for the past actions
            # Now past actions are T{t-h, t-h+1}, T{t-h+1, t-h+2}... T{t-1, t}
            # They will be inputted as T_{t-h, t-h+1}, T_{t-h, t-h+2}... T_{t-h, t}
            all_past_T_e_e_1 = [action_to_pose(action) for action in past_actions]
            for i in range(1, len(all_past_T_e_e_1)):
                all_past_T_e_e_1[i] = all_past_T_e_e_1[i - 1] @ all_past_T_e_e_1[i]

            past_actions = [pose_to_action(pose, gripper_opening, terminate) for pose, gripper_opening, terminate in
                            zip(all_past_T_e_e_1, past_actions[:, -2], past_actions[:, -1])]
        low_dim_action_labels = np.array(low_dim_action_labels)
        past_actions = np.array(past_actions)

        self.trans_rot_min = torch.minimum(self.trans_rot_min,
                                           torch.tensor(low_dim_action_labels[:, :6].min(axis=0)))
        self.trans_rot_max = torch.maximum(self.trans_rot_max,
                                           torch.tensor(low_dim_action_labels[:, :6].max(axis=0)))

        self.trans_rot_min_hist = torch.minimum(self.trans_rot_min_hist,
                                                torch.tensor(past_actions[:, :6].min(axis=0)))
        self.trans_rot_max_hist = torch.maximum(self.trans_rot_max_hist,
                                                torch.tensor(past_actions[:, :6].max(axis=0)))

        data = Data(
            pos=torch.tensor(pcd_eef, dtype=torch.float32),
            rgb=torch.tensor(pcd_rgb, dtype=torch.float32),
            actions=torch.tensor(np.array(low_dim_action_labels), dtype=torch.float32).unsqueeze(0),
            past_actions=torch.tensor(past_actions, dtype=torch.float32).unsqueeze(0),
            current_pose=torch.tensor(current_pose, dtype=torch.float32).unsqueeze(0),
            is_pad=torch.tensor(is_pad, dtype=torch.bool).unsqueeze(0),  # Not used
            text_features=text_features,
            interaction_trajectory=torch.tensor([interaction_trajectory]).unsqueeze(0)
        )

        self.chunk.append(data)

        if len(self.chunk) == self.dp_per_chunk:
            while True:
                try:
                    fdir = join(
                        self.processed_dir, f'data_{updated_dset_len}.pkl')
                    with open(fdir, 'xb', opener=opener) as f:
                        pickle.dump(self.chunk, f)
                    self.chunk = []
                    return updated_dset_len + 1
                except FileExistsError:
                    updated_dset_len += 1
        return updated_dset_len

    def process(self):

        def embed_task_name(tdir: str):
            # Text features for the task, uses the name of the directory.
            tdir = join(self.root, tdir)
            task_name = get_skill_name(tdir).lower()
            text_tokens = clip.tokenize([task_name.replace('_', ' ')])
            with torch.no_grad():
                return self.clip_model.encode_text(text_tokens).float()

        if not self.reprocess:
            return

        # Get dataset statistics to later normalise the data

        trj_counter = 0
        gen_done = False
        counter = self.mean_counter
        mean_counter = self.mean_counter

        trans_and_rot_mean = self.trans_rot_mean
        trans_and_rot_var = self.trans_rot_std.pow(2)
        demo_dps = 0
        bn_reaching_dps = 0
        term_actions = 0
        non_term_actions = 0
        open_gripper_actions = 0
        closed_gripper_actions = 0

        tn_feat = [
            embed_task_name(tdir) for tdir in self.task_dirs
        ]
        random_task_indeces = np.arange(len(tn_feat))
        np.random.shuffle(random_task_indeces)

        with tqdm(total=self.total_chunks_to_gen, desc='Dataset Generation', position=0) as pbar, \
                tqdm(total=self.dp_per_chunk, desc='Building Chunk ', position=1) as cbar:
            while True:
                pbar.reset()
                # print(self.dataset_length)
                pbar.update(self.dataset_length)  # accounts for multiple parallel processes
                pbar.refresh()

                task_idx = random_task_indeces[trj_counter % len(tn_feat)]
                trj_counter += 1
                task_dir, text_features = self.task_dirs[task_idx], tn_feat[task_idx]
                # If you want to consider just certain tasks, you can do it here.
                # if 'pick_up_white_pan' != task_dir:
                #     continue

                # Grab all data for a single demo.
                (_, rgb_init, depth_init, mask_init, _, _, _, traj_eef_posevec, _,
                 _, _, external_camera_intrinsics, bottleneck_pose) \
                    = self.load_demo_data(task_dir, False, True, True, True, False, False, False, True,
                                          False, False, False, True, True)

                # # Get point clouds in camera frame
                # print(f'Projecting depth to point clouds for: {task_dir}.')
                # traj_pcds_cam = []
                # traj_rgb_pcds = []
                # for traj_depth_curr, traj_rgb_curr, traj_mask_curr in zip(traj_depth, traj_rgb, traj_masks):
                #     pcd_cam_curr, pcd_rgb_curr = backproject_camera_target_realworld(im_depth=traj_depth_curr / 1000,
                #                                                                      rgb=traj_rgb_curr,
                #                                                                      K=external_camera_intrinsics,
                #                                                                      target_mask=np.logical_not(
                #                                                                          traj_mask_curr))
                #     pcd_cam_curr, pcd_rgb_curr = remove_small_clusters_from_pcd(pcd_cam_curr,
                #                                                                 pcd_rgb_curr,
                #                                                                 self.min_num_points_per_cluster)
                #     traj_pcds_cam.append(pcd_cam_curr)
                #     traj_rgb_pcds.append(pcd_rgb_curr)
                #
                # print('Done projecting depth to point clouds.')

                ########################################################################################################
                # Adding linear trajectories to the bottleneck pose.
                ########################################################################################################
                pcd_cam_initial, pcd_rgb_initial = backproject_camera_target_realworld(
                    im_depth=depth_init / 1000,
                    rgb=rgb_init,
                    K=external_camera_intrinsics,
                    target_mask=np.logical_not(mask_init))
                pcd_cam_initial, pcd_rgb_initial = remove_small_clusters_from_pcd(pcd_cam_initial,
                                                                                  pcd_rgb_initial,
                                                                                  self.min_num_points_per_cluster)

                # Grab initial pose and randomise it. Conversion from rotvec 2 rot only works as intended because the
                # randomisation range has only a single non-zero number.
                init_pose = self.neutral_pose.copy()
                init_pose[:3, 3] += np.random.uniform(self.trans_perturb_to_bn_lb, self.trans_perturb_to_bn_ub)
                init_pose[:3, :3] = rotvec2rot(
                    np.random.uniform(self.rot_perturb_to_bn_lb, self.rot_perturb_to_bn_ub)) @ init_pose[:3, :3]

                # Interpolate the current pose to the bottleneck pose to get the actions.
                num_points_in_path_to_bn = np.linalg.norm(
                    init_pose[:3, 3] - bottleneck_pose[:3, 3]) / self.trans_spacing_between_waypoints
                num_points_in_path_to_bn = int(np.ceil(num_points_in_path_to_bn)) + 1  # + 1 as first pose is init_pose

                # After interpolation, you'll need to add an extra corrective action.
                bottleneck_reaching_eef_poses = interpolate_poses(init_pose, bottleneck_pose, num_points_in_path_to_bn)

                # Just calculating the deltas between the poses. Always reaching to the bottleneck with closed gripper
                # -- action 1.
                # TODO:
                low_dim_eef_bottleneck_reaching_actions = [
                    pose_to_action(
                        pose=np.linalg.inv(bottleneck_reaching_eef_poses[k]) @ bottleneck_reaching_eef_poses[k + 1],
                        gripper_opening=self.closed_gripper_action,
                        terminate=self.not_terminate_action) for k in range(num_points_in_path_to_bn - 1)]
                ###
                # Adding action to terminate at the end of each trajectory. Only is here for the BN reaching network
                low_dim_eef_bottleneck_reaching_actions.append(pose_to_action(pose=np.eye(4),
                                                                              gripper_opening=traj_eef_posevec[-1][-1],
                                                                              terminate=self.terminate_action))
                bottleneck_reaching_eef_poses.append(bottleneck_reaching_eef_poses[-1])
                ###

                low_dim_eef_bottleneck_reaching_actions = np.array(low_dim_eef_bottleneck_reaching_actions)

                # Update counters
                # demo_dps += len(low_dim_eef_traj_actions)
                bn_reaching_dps += len(low_dim_eef_bottleneck_reaching_actions)

                # bottleneck_idx is redundant here
                # all_low_dim_eef_actions = np.concatenate((low_dim_eef_bottleneck_reaching_actions,
                #                                           low_dim_eef_traj_actions[bottleneck_idx:]), axis=0)
                all_low_dim_eef_actions = low_dim_eef_bottleneck_reaching_actions
                all_eef_poses = bottleneck_reaching_eef_poses
                # all_eef_poses = np.concatenate((bottleneck_reaching_eef_poses[:-1], traj_eef_poses[bottleneck_idx:]), axis=0)

                # action_history = np.array(
                #     [[0., 0., 0., 0., 0., 0., self.closed_gripper_action,
                #       self.not_terminate_action]] * self.history_size)
                action_history = np.array([[0., 0., 0., 0., 0., 0., self.not_terminate_action]] * self.history_size)

                # For history normalization
                all_past_actions = np.zeros((len(all_low_dim_eef_actions), self.history_size, 6))

                # We have actions, now getting (state, action) pairs.
                for timestep_num in range(len(all_low_dim_eef_actions)):

                    current_pose = all_eef_poses[timestep_num]
                    # This is because, np.concatenate doesn't work with 0d arrays.
                    if timestep_num != 0:
                        past_actions = np.concatenate((
                            action_history[timestep_num:],
                            np.array(all_low_dim_eef_actions[max(0, timestep_num - self.history_size): timestep_num])),
                            axis=0)
                    else:
                        past_actions = action_history[timestep_num:]

                    # For normalising the history
                    all_past_actions[timestep_num] = past_actions[..., :6]

                    low_dim_action_labels = np.array(
                        all_low_dim_eef_actions[timestep_num:self.chunk_size + timestep_num])

                    # Pad if it goes over the end of the demo.
                    is_pad = np.zeros(self.chunk_size)
                    if len(low_dim_action_labels) < self.chunk_size:
                        is_pad[-(len(is_pad) - len(low_dim_action_labels)):] = 0  # if 1., then would pad the end.
                        # low_dim_action_labels = np.concatenate((
                        #     low_dim_action_labels,
                        #     np.array((self.chunk_size - len(low_dim_action_labels)) * [
                        #         [0., 0., 0., 0., 0., 0., low_dim_eef_traj_actions[-1][-2], self.terminate_action]])))
                        low_dim_action_labels = np.concatenate((
                            low_dim_action_labels,
                            np.array((self.chunk_size - len(low_dim_action_labels)) * [
                                [0., 0., 0., 0., 0., 0., self.terminate_action]])))

                    # If we are already on a trajectory, we need to use correct observations.
                    if timestep_num >= len(low_dim_eef_bottleneck_reaching_actions):
                        raise Exception('\n\nSomething went wrong. We should never be here.\n\n')  # TODO remove
                        pcd_cam = traj_pcds_cam[timestep_num - len(low_dim_eef_bottleneck_reaching_actions)]
                        pcd_rgb = traj_rgb_pcds[timestep_num - len(low_dim_eef_bottleneck_reaching_actions)]
                        # Expressing the pcd in the EEF frame. # TODO: should this be the tip frame?
                        pcd_eef = transform_pcd_np(pcd_cam, np.linalg.inv(current_pose) @ self.T_WC)
                        interaction_trajectory = True
                    else:
                        # Expressing the pcd in the EEF frame.  # TODO: should this be the tip frame?
                        pcd_eef = transform_pcd_np(pcd_cam_initial, np.linalg.inv(current_pose) @ self.T_WC)
                        pcd_rgb = pcd_rgb_initial
                        interaction_trajectory = False

                    # downsample the point cloud
                    pcd_eef, pcd_rgb = downsample_pcd(pcd_eef, pcd_rgb, self.num_points_to_save)

                    # This doesn't go through the network. It is only used for visualisation
                    current_pose = pose_to_action(
                        pose=current_pose,  # pose=np.linalg.inv(current_pose) @ all_eef_poses[timestep_num + 1],
                        gripper_opening=all_low_dim_eef_actions[timestep_num][-2],
                        terminate=all_low_dim_eef_actions[timestep_num][-1])

                    # gripper_opening=low_dim_eef_bottleneck_reaching_actions[0][-2],
                    # terminate=low_dim_eef_bottleneck_reaching_actions[0][-1])

                    # interaction_part = timestep_num >

                    # # Creating PyTorch Geometric Data object
                    # self.save_data_point(counter, pcd_eef, pcd_rgb, low_dim_action_labels, past_actions,
                    #                      current_pose, text_features, interaction_trajectory)

                    cbar.update(1)
                    updated_dset_len = self.save_data_point(pcd_eef, pcd_rgb, low_dim_action_labels, past_actions,
                                                            current_pose, is_pad, text_features, interaction_trajectory)
                    self.dataset_length = updated_dset_len
                    if len(self.chunk) == 0:
                        cbar.reset()
                    if self.dataset_length >= self.total_chunks_to_gen:
                        gen_done = True
                        break
                    counter += 1
                if gen_done:
                    break

                # Generate X% of corrective actions in the proximity of the bottleneck pose
                for _ in range(int(np.ceil(self.corrective_action_ratio * len(all_low_dim_eef_actions)))):
                    # Sample noise vector
                    trans_noise = np.random.randn(3)
                    trans_noise = trans_noise / np.linalg.norm(trans_noise)
                    trans_noise = trans_noise * np.random.uniform(1e-3, self.trans_spacing_between_waypoints)

                    noise_rotvec = np.random.randn(3)
                    noise_rotvec = np.random.uniform(np.deg2rad(0.5), np.deg2rad(5)) * noise_rotvec / np.linalg.norm(
                        noise_rotvec)
                    R_noise = rotvec2rot(noise_rotvec)

                    # Perturb bottleneck pose
                    current_pose = bottleneck_pose.copy()
                    current_pose[:3, :3] = R_noise @ current_pose[:3, :3]
                    current_pose[:3, 3] = trans_noise + current_pose[:3, 3]

                    corrective_action = np.array(
                        [pose_to_action(pose=np.linalg.inv(current_pose) @ bottleneck_pose,
                                        gripper_opening=self.closed_gripper_action,
                                        terminate=self.terminate_action)])

                    corrective_action = np.concatenate(
                        (corrective_action, np.array((self.chunk_size - len(corrective_action)) * [
                            [0., 0., 0., 0., 0., 0., self.terminate_action]])))

                    # Expressing the pcd in the EEF frame.  # TODO: should this be the tip frame?
                    pcd_eef = transform_pcd_np(pcd_cam_initial, np.linalg.inv(current_pose) @ self.T_WC)
                    pcd_rgb = pcd_rgb_initial
                    interaction_trajectory = False

                    # downsample the point cloud
                    pcd_eef, pcd_rgb = downsample_pcd(pcd_eef, pcd_rgb, self.num_points_to_save)

                    # This doesn't go through the network. It is only used for visualisation
                    current_pose = pose_to_action(
                        pose=current_pose,
                        gripper_opening=self.closed_gripper_action,
                        terminate=self.not_terminate_action)

                    cbar.update(1)
                    updated_dset_len = self.save_data_point(pcd_eef, pcd_rgb, corrective_action, past_actions,
                                                            current_pose, is_pad, text_features, interaction_trajectory)
                    self.dataset_length = updated_dset_len
                    if len(self.chunk) == 0:
                        cbar.reset()
                    if self.dataset_length >= self.total_chunks_to_gen:
                        gen_done = True
                        break
                    counter += 1
                if gen_done:
                    break
                ########################################################################################################
                # Perform a batch update to mean and variance
                # (see https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html)
                n = mean_counter
                mu_n = trans_and_rot_mean.clone()
                var_n = trans_and_rot_var.clone()

                m = len(all_low_dim_eef_actions[:, :6])
                mu_m = torch.tensor(all_low_dim_eef_actions[:, :6].mean(axis=0))
                var_m = torch.tensor(all_low_dim_eef_actions[:, :6].var(axis=0))

                trans_and_rot_mean = (m / (n + m)) * mu_m + (n / (n + m)) * mu_n
                trans_and_rot_var = (m / (n + m)) * var_m + (n / (n + m)) * var_n + ((n * m) / (m + n) ** 2) * (
                        mu_m - mu_n) ** 2

                term_actions += (all_low_dim_eef_actions[:, -1] == self.terminate_action).sum()
                non_term_actions += (all_low_dim_eef_actions[:, -1] == self.not_terminate_action).sum()
                # open_gripper_actions += (all_low_dim_eef_actions[:, 6] == self.open_gripper_action).sum()
                # closed_gripper_actions += (all_low_dim_eef_actions[: 6] == self.closed_gripper_action).sum()

                mean_counter = counter
                ########################################################################################################
                ########################################################################################################

                ########################################################################################################
                if trj_counter % 30 == 0:
                    torch.save(mean_counter, join(self.processed_dir, 'mean_counter.pt'))
                    self.trans_rot_mean = trans_and_rot_mean
                    self.trans_rot_std = trans_and_rot_var.sqrt()
                    # self.trans_rot_min = trans_and_rot_min
                    # self.trans_rot_max = trans_and_rot_max
                    # self.trans_rot_min_hist = trans_and_rot_min_hist
                    # self.trans_rot_max_hist = trans_and_rot_max_hist
                    torch.save(self.trans_rot_mean, join(self.processed_dir, 'trans_and_rot_mean.pt'))
                    torch.save(self.trans_rot_std, join(self.processed_dir, 'trans_and_rot_std.pt'))

                    # To avoid devision by zero we +/- 1e-6
                    torch.save(self.trans_rot_min - 1e-6, join(self.processed_dir, 'trans_and_rot_min.pt'))
                    torch.save(self.trans_rot_max + 1e-6, join(self.processed_dir, 'trans_and_rot_max.pt'))
                    torch.save(self.trans_rot_min_hist - 1e-6, join(self.processed_dir, 'trans_and_rot_min_hist.pt'))
                    torch.save(self.trans_rot_max_hist + 1e-6, join(self.processed_dir, 'trans_and_rot_max_hist.pt'))

                    self.num_bn_pts += bn_reaching_dps
                    self.num_demo_pts += demo_dps
                    self.num_term_acts += term_actions
                    self.num_dont_term_acts += non_term_actions
                    self.num_open_acts += open_gripper_actions
                    self.num_close_acts += closed_gripper_actions

                    demo_dps = 0
                    bn_reaching_dps = 0
                    term_actions = 0
                    non_term_actions = 0
                    open_gripper_actions = 0
                    closed_gripper_actions = 0

                    with open(join(self.processed_dir, 'statistics.txt'), 'w', opener=opener) as f:
                        f.write(f'Dataset length: {self.dataset_length * self.dp_per_chunk}\n')
                        f.write(f'Total number of points for bottleneck reaching: {self.num_bn_pts}\n')
                        f.write(f'Total number of points for interactions: {self.num_demo_pts}\n')
                        f.write(f'Ratio: {self.num_bn_pts / (self.num_bn_pts + self.num_demo_pts)}\n')
                        f.write('\n')
                        f.write(f'Total number of terminate actions: {self.num_term_acts}\n')
                        f.write(f'Total number of don\'t terminate actions: {self.num_dont_term_acts}\n')
                        f.write('\n')
                        f.write(f'Total number of open gripper actions: {self.num_open_acts}\n')
                        f.write(f'Total number of close gripper actions: {self.num_close_acts}\n')
                        f.write('\n')

    # def process(self):

    #     if not self.reprocess:
    #         return

    #     # Get dataset statistics to later normalise the data

    #     counter = 0
    #     mean_counter = 0
    #     trans_and_rot_min = torch.inf * torch.ones(6)
    #     trans_and_rot_max = - torch.inf * torch.ones(6)

    #     trans_and_rot_min_hist = torch.inf * torch.ones(6)
    #     trans_and_rot_max_hist = - torch.inf * torch.ones(6)

    #     trans_and_rot_mean = torch.zeros(6)
    #     trans_and_rot_var = torch.zeros(6)
    #     demo_dps = 0
    #     bn_reaching_dps = 0
    #     term_actions = 0
    #     non_term_actions = 0
    #     open_gripper_actions = 0
    #     closed_gripper_actions = 0
    #     for task_dir in self.task_dirs:
    #         # If you want to consider just certain tasks, you can do it here.
    #         # if 'pick_up_white_pan' != task_dir:
    #         #     continue

    #         # Grab all data for a single demo.
    #         (task_name, rgb_init, depth_init, mask_init, traj_rgb, traj_depth, traj_masks, traj_eef_posevec,
    #          traj_eef_poses,
    #          low_dim_eef_traj_actions, bottleneck_idx, external_camera_intrinsics, bottleneck_pose) \
    #             = self.load_demo_data(task_dir)

    #         # # Get point clouds in camera frame
    #         # print(f'Projecting depth to point clouds for: {task_dir}.')
    #         # traj_pcds_cam = []
    #         # traj_rgb_pcds = []
    #         # for traj_depth_curr, traj_rgb_curr, traj_mask_curr in zip(traj_depth, traj_rgb, traj_masks):
    #         #     pcd_cam_curr, pcd_rgb_curr = backproject_camera_target_realworld(im_depth=traj_depth_curr / 1000,
    #         #                                                                      rgb=traj_rgb_curr,
    #         #                                                                      K=external_camera_intrinsics,
    #         #                                                                      target_mask=np.logical_not(
    #         #                                                                          traj_mask_curr))
    #         #     pcd_cam_curr, pcd_rgb_curr = remove_small_clusters_from_pcd(pcd_cam_curr,
    #         #                                                                 pcd_rgb_curr,
    #         #                                                                 self.min_num_points_per_cluster)
    #         #     traj_pcds_cam.append(pcd_cam_curr)
    #         #     traj_rgb_pcds.append(pcd_rgb_curr)
    #         #
    #         # print('Done projecting depth to point clouds.')

    #         # Text features for the task, uses the name of the directory.
    #         text_tokens = clip.tokenize([task_name.replace('_', ' ')])
    #         with torch.no_grad():
    #             text_features = self.clip_model.encode_text(text_tokens).float()

    #         ########################################################################################################
    #         # Adding linear trajectories to the bottleneck pose.
    #         ########################################################################################################
    #         pcd_cam_initial, pcd_rgb_initial = backproject_camera_target_realworld(
    #             im_depth=depth_init / 1000,
    #             rgb=rgb_init,
    #             K=external_camera_intrinsics,
    #             target_mask=np.logical_not(mask_init))
    #         pcd_cam_initial, pcd_rgb_initial = remove_small_clusters_from_pcd(pcd_cam_initial,
    #                                                                           pcd_rgb_initial,
    #                                                                           self.min_num_points_per_cluster)

    #         for _ in tqdm(range(self.num_traj_to_bn), desc=f'Processing dataset, {task_dir}, extra', leave=False):

    #             # Grab initial pose and randomise it. Conversion from rotvec 2 rot only works as intended because the
    #             # randomisation range has only a single non-zero number.
    #             init_pose = self.neutral_pose.copy()
    #             init_pose[:3, 3] += np.random.uniform(self.trans_perturb_to_bn_lb, self.trans_perturb_to_bn_ub)
    #             init_pose[:3, :3] = rotvec2rot(
    #                 np.random.uniform(self.rot_perturb_to_bn_lb, self.rot_perturb_to_bn_ub)) @ init_pose[:3, :3]

    #             # Interpolate the current pose to the bottleneck pose to get the actions.
    #             num_points_in_path_to_bn = np.linalg.norm(
    #                 init_pose[:3, 3] - bottleneck_pose[:3, 3]) / self.trans_spacing_between_waypoints
    #             num_points_in_path_to_bn = int(np.ceil(num_points_in_path_to_bn)) + 1  # + 1 as first pose is init_pose

    #             # After interpolation, you'll need to add an extra corrective action.
    #             bottleneck_reaching_eef_poses = interpolate_poses(init_pose, bottleneck_pose, num_points_in_path_to_bn)

    #             # Just calculating the deltas between the poses. Always reaching to the bottleneck with closed gripper
    #             # -- action 1.
    #             # TODO:
    #             low_dim_eef_bottleneck_reaching_actions = [
    #                 pose_to_action(
    #                     pose=np.linalg.inv(bottleneck_reaching_eef_poses[k]) @ bottleneck_reaching_eef_poses[k + 1],
    #                     gripper_opening=self.closed_gripper_action,
    #                     terminate=self.not_terminate_action) for k in range(num_points_in_path_to_bn - 1)]
    #             ###
    #             # Adding action to terminate at the end of each trajectory. Only is here for the BN reaching network
    #             low_dim_eef_bottleneck_reaching_actions.append(pose_to_action(pose=np.eye(4),
    #                                                                           gripper_opening=traj_eef_posevec[-1][-1],
    #                                                                           terminate=self.terminate_action))
    #             bottleneck_reaching_eef_poses.append(bottleneck_reaching_eef_poses[-1])
    #             ###

    #             low_dim_eef_bottleneck_reaching_actions = np.array(low_dim_eef_bottleneck_reaching_actions)

    #             # Update counters
    #             # demo_dps += len(low_dim_eef_traj_actions)
    #             bn_reaching_dps += len(low_dim_eef_bottleneck_reaching_actions)

    #             # bottleneck_idx is redundant here
    #             # all_low_dim_eef_actions = np.concatenate((low_dim_eef_bottleneck_reaching_actions,
    #             #                                           low_dim_eef_traj_actions[bottleneck_idx:]), axis=0)
    #             all_low_dim_eef_actions = low_dim_eef_bottleneck_reaching_actions
    #             all_eef_poses = bottleneck_reaching_eef_poses
    #             # all_eef_poses = np.concatenate((bottleneck_reaching_eef_poses[:-1], traj_eef_poses[bottleneck_idx:]), axis=0)

    #             # action_history = np.array(
    #             #     [[0., 0., 0., 0., 0., 0., self.closed_gripper_action,
    #             #       self.not_terminate_action]] * self.history_size)
    #             action_history = np.array([[0., 0., 0., 0., 0., 0., self.not_terminate_action]] * self.history_size)

    #             # For history normalization
    #             all_past_actions = np.zeros((len(all_low_dim_eef_actions), self.history_size, 6))

    #             # We have actions, now getting (state, action) pairs.
    #             for timestep_num in range(len(all_low_dim_eef_actions)):

    #                 current_pose = all_eef_poses[timestep_num]
    #                 # This is because, np.concatenate doesn't work with 0d arrays.
    #                 if timestep_num != 0:
    #                     past_actions = np.concatenate((
    #                         action_history[timestep_num:],
    #                         np.array(all_low_dim_eef_actions[max(0, timestep_num - self.history_size): timestep_num])),
    #                         axis=0)
    #                 else:
    #                     past_actions = action_history[timestep_num:]

    #                 # For normalising the history
    #                 all_past_actions[timestep_num] = past_actions[..., :6]

    #                 low_dim_action_labels = np.array(
    #                     all_low_dim_eef_actions[timestep_num:self.chunk_size + timestep_num])

    #                 # Pad if it goes over the end of the demo.
    #                 is_pad = np.zeros(self.chunk_size)
    #                 if len(low_dim_action_labels) < self.chunk_size:
    #                     is_pad[-(len(is_pad) - len(low_dim_action_labels)):] = 0  # if 1., then would pad the end.
    #                     # low_dim_action_labels = np.concatenate((
    #                     #     low_dim_action_labels,
    #                     #     np.array((self.chunk_size - len(low_dim_action_labels)) * [
    #                     #         [0., 0., 0., 0., 0., 0., low_dim_eef_traj_actions[-1][-2], self.terminate_action]])))
    #                     low_dim_action_labels = np.concatenate((
    #                         low_dim_action_labels,
    #                         np.array((self.chunk_size - len(low_dim_action_labels)) * [
    #                             [0., 0., 0., 0., 0., 0., self.terminate_action]])))

    #                 # If we are already on a trajectory, we need to use correct observations.
    #                 if timestep_num >= len(low_dim_eef_bottleneck_reaching_actions):
    #                     raise Exception('\n\nSomething went wrong. We should never be here.\n\n')  # TODO remove
    #                     pcd_cam = traj_pcds_cam[timestep_num - len(low_dim_eef_bottleneck_reaching_actions)]
    #                     pcd_rgb = traj_rgb_pcds[timestep_num - len(low_dim_eef_bottleneck_reaching_actions)]
    #                     # Expressing the pcd in the EEF frame. # TODO: should this be the tip frame?
    #                     pcd_eef = transform_pcd_np(pcd_cam, np.linalg.inv(current_pose) @ self.T_WC)
    #                     interaction_trajectory = True
    #                 else:
    #                     # Expressing the pcd in the EEF frame.  # TODO: should this be the tip frame?
    #                     pcd_eef = transform_pcd_np(pcd_cam_initial, np.linalg.inv(current_pose) @ self.T_WC)
    #                     pcd_rgb = pcd_rgb_initial
    #                     interaction_trajectory = False

    #                 # downsample the point cloud
    #                 pcd_eef, pcd_rgb = downsample_pcd(pcd_eef, pcd_rgb, self.num_points_to_save)

    #                 # This doesn't go through the network. It is only used for visualisation
    #                 current_pose = pose_to_action(
    #                     pose=current_pose,  # pose=np.linalg.inv(current_pose) @ all_eef_poses[timestep_num + 1],
    #                     gripper_opening=all_low_dim_eef_actions[timestep_num][-2],
    #                     terminate=all_low_dim_eef_actions[timestep_num][-1])

    #                 # gripper_opening=low_dim_eef_bottleneck_reaching_actions[0][-2],
    #                 # terminate=low_dim_eef_bottleneck_reaching_actions[0][-1])

    #                 # interaction_part = timestep_num >

    #                 # # Creating PyTorch Geometric Data object
    #                 # self.save_data_point(counter, pcd_eef, pcd_rgb, low_dim_action_labels, past_actions,
    #                 #                      current_pose, text_features, interaction_trajectory)
    #                 self.save_data_point(counter, pcd_eef, pcd_rgb, low_dim_action_labels, past_actions,
    #                                      current_pose, is_pad, text_features, interaction_trajectory)
    #                 counter += 1

    #             # Generate X% of corrective actions in the proximity of the bottleneck pose
    #             for _ in range(int(np.ceil(self.corrective_action_ratio * len(all_low_dim_eef_actions)))):
    #                 # Sample noise vector
    #                 trans_noise = np.random.randn(3)
    #                 trans_noise = trans_noise / np.linalg.norm(trans_noise)
    #                 trans_noise = trans_noise * np.random.uniform(1e-3, self.trans_spacing_between_waypoints)

    #                 noise_rotvec = np.random.randn(3)
    #                 noise_rotvec = np.random.uniform(np.deg2rad(0.5), np.deg2rad(5)) * noise_rotvec / np.linalg.norm(
    #                     noise_rotvec)
    #                 R_noise = rotvec2rot(noise_rotvec)

    #                 # Perturb bottleneck pose
    #                 current_pose = bottleneck_pose.copy()
    #                 current_pose[:3, :3] = R_noise @ current_pose[:3, :3]
    #                 current_pose[:3, 3] = trans_noise + current_pose[:3, 3]

    #                 corrective_action = np.array(
    #                     [pose_to_action(pose=np.linalg.inv(current_pose) @ bottleneck_pose,
    #                                     gripper_opening=self.closed_gripper_action,
    #                                     terminate=self.terminate_action)])

    #                 corrective_action = np.concatenate(
    #                     (corrective_action, np.array((self.chunk_size - len(corrective_action)) * [
    #                         [0., 0., 0., 0., 0., 0., self.terminate_action]])))

    #                 # Expressing the pcd in the EEF frame.  # TODO: should this be the tip frame?
    #                 pcd_eef = transform_pcd_np(pcd_cam_initial, np.linalg.inv(current_pose) @ self.T_WC)
    #                 pcd_rgb = pcd_rgb_initial
    #                 interaction_trajectory = False

    #                 # downsample the point cloud
    #                 pcd_eef, pcd_rgb = downsample_pcd(pcd_eef, pcd_rgb, self.num_points_to_save)

    #                 # This doesn't go through the network. It is only used for visualisation
    #                 current_pose = pose_to_action(
    #                     pose=current_pose,
    #                     gripper_opening=self.closed_gripper_action,
    #                     terminate=self.not_terminate_action)

    #                 self.save_data_point(counter, pcd_eef, pcd_rgb, corrective_action, past_actions,
    #                                      current_pose, is_pad, text_features, interaction_trajectory)
    #                 counter += 1
    #             ########################################################################################################
    #             # Perform a batch update to mean and variance
    #             # (see https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html)
    #             n = mean_counter
    #             mu_n = trans_and_rot_mean.clone()
    #             var_n = trans_and_rot_var.clone()

    #             m = len(all_low_dim_eef_actions[:, :6])
    #             mu_m = torch.tensor(all_low_dim_eef_actions[:, :6].mean(axis=0))
    #             var_m = torch.tensor(all_low_dim_eef_actions[:, :6].var(axis=0))

    #             trans_and_rot_mean = (m / (n + m)) * mu_m + (n / (n + m)) * mu_n
    #             trans_and_rot_var = (m / (n + m)) * var_m + (n / (n + m)) * var_n + ((n * m) / (m + n) ** 2) * (
    #                     mu_m - mu_n) ** 2

    #             # Update min and max value
    #             trans_and_rot_min = torch.minimum(trans_and_rot_min,
    #                                               torch.tensor(all_low_dim_eef_actions[:, :6].min(axis=0)))
    #             trans_and_rot_max = torch.maximum(trans_and_rot_max,
    #                                               torch.tensor(all_low_dim_eef_actions[:, :6].max(axis=0)))

    #             # Update min and max value for the history
    #             all_past_actions = all_past_actions.reshape(-1, 6)
    #             trans_and_rot_min_hist = torch.minimum(trans_and_rot_min_hist,
    #                                                    torch.tensor(all_past_actions[:, :6].min(axis=0)))
    #             trans_and_rot_max_hist = torch.maximum(trans_and_rot_max_hist,
    #                                                    torch.tensor(all_past_actions[:, :6].max(axis=0)))

    #             term_actions += (all_low_dim_eef_actions[:, -1] == self.terminate_action).sum()
    #             non_term_actions += (all_low_dim_eef_actions[:, -1] == self.not_terminate_action).sum()
    #             # open_gripper_actions += (all_low_dim_eef_actions[:, 6] == self.open_gripper_action).sum()
    #             # closed_gripper_actions += (all_low_dim_eef_actions[: 6] == self.closed_gripper_action).sum()

    #             mean_counter = counter
    #         ########################################################################################################
    #         ########################################################################################################

    #     ########################################################################################################
    #     self.dataset_length = counter
    #     self.trans_rot_mean = trans_and_rot_mean
    #     self.trans_rot_std = trans_and_rot_var.sqrt()
    #     # self.trans_rot_min = trans_and_rot_min
    #     # self.trans_rot_max = trans_and_rot_max
    #     # self.trans_rot_min_hist = trans_and_rot_min_hist
    #     # self.trans_rot_max_hist = trans_and_rot_max_hist
    #     torch.save(self.trans_rot_mean, join(self.processed_dir, 'trans_and_rot_mean.pt'))
    #     torch.save(self.trans_rot_std, join(self.processed_dir, 'trans_and_rot_std.pt'))

    #     # To avoid devision by zero
    #     self.trans_rot_min -= 1e-6
    #     self.trans_rot_max += 1e-6
    #     self.trans_rot_min_hist -= 1e-6
    #     self.trans_rot_max_hist += 1e-6

    #     torch.save(self.trans_rot_min, join(self.processed_dir, 'trans_and_rot_min.pt'))
    #     torch.save(self.trans_rot_max, join(self.processed_dir, 'trans_and_rot_max.pt'))
    #     torch.save(self.trans_rot_min_hist, join(self.processed_dir, 'trans_and_rot_min_hist.pt'))
    #     torch.save(self.trans_rot_max_hist, join(self.processed_dir, 'trans_and_rot_max_hist.pt'))

    #     self.num_bn_pts = bn_reaching_dps
    #     self.num_demo_pts = demo_dps
    #     self.num_term_acts = term_actions
    #     self.num_dont_term_acts = non_term_actions
    #     self.num_open_acts = open_gripper_actions
    #     self.num_close_acts = closed_gripper_actions

    #     with open(join(self.processed_dir, 'statistics.txt'), 'w') as f:
    #         f.write(f'Dataset length: {counter}\n')
    #         f.write(f'Total number of points for bottleneck reaching: {bn_reaching_dps}\n')
    #         f.write(f'Total number of points for interactions: {demo_dps}\n')
    #         f.write(f'Ratio: {bn_reaching_dps / (bn_reaching_dps + demo_dps)}\n')
    #         f.write('\n')
    #         f.write(f'Total number of terminate actions: {term_actions}\n')
    #         f.write(f'Total number of don\'t terminate actions: {non_term_actions}\n')
    #         f.write('\n')
    #         f.write(f'Total number of open gripper actions: {open_gripper_actions}\n')
    #         f.write(f'Total number of close gripper actions: {closed_gripper_actions}\n')
    #         f.write('\n')

    def len(self):
        return self.dataset_length
    
    def load_chunks(self, chunk_ids: torch.Tensor):
        self.loaded_chunks = []
        for cid in chunk_ids:
            loaded = False
            while not loaded:
                fpath = join(self.processed_dir, f'data_{cid}.pkl')
                try:
                    with open(fpath, 'rb') as f:
                        chunk = pickle.load(f)
                        self.loaded_chunks += chunk
                    loaded = True
                except (EOFError, FileNotFoundError, pickle.UnpicklingError) as e:
                    print(f"[Chunk {cid} Error]: {e}")
                    cid = np.random.randint(0, self._n_chunks)

    def get(self, idx):
        assert self.loaded_chunks is not None, "[BCDataset] You have to load the chunks first"
        # chunk_iter = int(idx / self.batch_size)
        # batch_iter = idx % self.batch_size
        # data = self.loaded_chunks[batch_iter][chunk_iter]
        data = self.loaded_chunks[idx]
        pcd_pos, pcd_rgb = data.pos, data.rgb

        if self.mask_pcd_clusters:
            # Mask out portions of the point cloud
            indices = fps(pcd_pos, ratio=self.total_num_clusters / len(pcd_pos))
            cluster_idx = nearest(pcd_pos,
                                  pcd_pos[indices])  # For each point, this gives the index of the closest centroid

            num_clusters_to_mask = np.random.randint(1, self.num_clusters_to_mask + 1)
            cluster_indices_to_keep = np.random.choice([cls_idx for cls_idx in range(self.total_num_clusters)],
                                                       size=self.total_num_clusters - num_clusters_to_mask,
                                                       replace=False)

            keep = np.isin(cluster_idx, cluster_indices_to_keep)
            pcd_pos, pcd_rgb = pcd_pos[keep], pcd_rgb[keep]

        # Down sample
        pcd_pos, pcd_rgb = downsample_pcd(pcd_pos, pcd_rgb, self.max_num_points)

        # Add gaussian noise to point cloud
        if self.add_noise_to_pcd:
            pcd_pos = pcd_pos + torch.randn_like(pcd_pos) * self.pcd_noise_std

        data.pos, data.rgb = pcd_pos, pcd_rgb

        # Scaling of actions and action history
        data.actions = action_scaling(data.actions,
                                      trans_rot_min=self.trans_rot_min,
                                      trans_rot_max=self.trans_rot_max)
        data.past_actions = action_scaling(data.past_actions,
                                           trans_rot_min=self.trans_rot_min_hist,
                                           trans_rot_max=self.trans_rot_max_hist)

        # data.actions[:, :, :6] = (data.actions[:, :, :6] - self.trans_and_rot_mean) / self.trans_and_rot_std
        # data.past_actions[:, :, :6] = (data.past_actions[:, :, :6] - self.trans_and_rot_mean) / self.trans_and_rot_std

        # Add noise to history
        if self.add_noise_to_act_hist:
            data.past_actions[:, :, :6] += self.act_hist_noise_std * torch.randn_like(data.past_actions[:, :, :6])

        ########################################################################################################
        if self.visualise:
            pcd = data.pos.numpy()
            T_WE = action_to_pose(data.current_pose.numpy()[0])
            pcd = transform_pcd_np(pcd, T_WE)  # pcd in world frame
            self.plotter.add_mesh(pcd, color='blue', point_size=10, render_points_as_spheres=True, name='pcd',
                                  opacity=1.)
            draw_coord_frame(self.plotter, T_WE, name='T_WE')
            # draw_coord_frame(self.plotter, np.eye(4), name='World')

            gt_past_actions = data.past_actions
            gt_past_actions = action_unscaling(gt_past_actions,
                                               tgt_trans_rot_min=self.trans_rot_min_hist,
                                               tgt_trans_rot_max=self.trans_rot_max_hist)
            # gt_past_actions[..., :6] = (gt_past_actions[..., :6] * self.trans_and_rot_std) + self.trans_and_rot_mean
            gt_past_actions = gt_past_actions.squeeze(0).cpu().numpy()

            gt_actions = data.actions
            gt_actions = action_unscaling(gt_actions,
                                          tgt_trans_rot_min=self.trans_rot_min,
                                          tgt_trans_rot_max=self.trans_rot_max)
            # gt_actions[..., :6] = (gt_actions[..., :6] * self.trans_and_rot_std) + self.trans_and_rot_mean
            gt_actions = gt_actions.squeeze(0).cpu().numpy()

            poses_gt = [T_WE]
            poses_gt_past = [T_WE]

            if gt_actions[0, -2] < 0.5:
                self.plotter.add_text('Open Gripper (t + 1)', position='lower_right', font_size=20, name='gripper')
            else:
                self.plotter.add_text('Close Gripper (t + 1)', position='lower_right', font_size=20, name='gripper')

            if gt_actions[0, -1] < 0.5:
                self.plotter.add_text('Continue (t + 1)', position='lower_left', font_size=20, name='terminate')
            else:
                self.plotter.add_text('Terminate (t + 1)', position='lower_left', font_size=20, name='terminate')

            if data.interaction_trajectory:
                self.plotter.add_text('Fine', position='upper_edge', font_size=20, name='act_interaction')
            else:
                self.plotter.add_text('Coarse', position='upper_edge', font_size=20, name='act_interaction')

            for j in range(len(gt_actions)):
                if self.action_mode == 'abs_delta':
                    poses_gt.append(poses_gt[0] @ action_to_pose(gt_actions[j]))
                else:
                    poses_gt.append(poses_gt[-1] @ action_to_pose(gt_actions[j]))

            if self.action_mode == 'abs_delta':
                # Change back to delta: T_{t-h, t-h+1} -> T_{t-h-1, t-h}
                all_past_T_eh_et = [action_to_pose(action) for action in gt_past_actions]
                new_poses = [all_past_T_eh_et[0]]
                for i in range(1, len(all_past_T_eh_et)):
                    new_poses.append(np.linalg.inv(all_past_T_eh_et[i - 1]) @ all_past_T_eh_et[i])
                gt_past_actions = [pose_to_action(pose, gripper_opening, terminate) for
                                   pose, gripper_opening, terminate in
                                   zip(new_poses, np.array(gt_past_actions)[:, -2], np.array(gt_past_actions)[:, -1])]

            for j in range(len(gt_past_actions)):
                poses_gt_past.append(poses_gt_past[-1] @ np.linalg.inv(action_to_pose(gt_past_actions[-j - 1])))

            for k, T_WE in enumerate(poses_gt):
                draw_coord_frame(self.plotter, poses_gt[k], name=f'T_WE_gt_{k}', opacity=0.3)

            for k, T_WE in enumerate(poses_gt_past):
                draw_coord_frame(self.plotter, poses_gt_past[k], name=f'T_WE_gt_past_{k}', opacity=0.3, scale=0.5)

            if not self.shown:
                self.plotter.show(auto_close=False)
                self.shown = True
        ########################################################################################################
        return data


if __name__ == '__main__':
    import socket
    from thousand_tasks.training.act_bn_reaching.config import mt_act_args_dict

    pc_name = socket.gethostname()

    if pc_name == 'slifer':
        tasks_dir_path = '/home/kamil/phd/head_cam/assets/tasks'
    elif pc_name == 'tiger' or pc_name == 'omen':
        tasks_dir_path = '/home/kamil/phd/osil/assets/tasks_rot'
    elif pc_name == 'Kamils-Air':
        tasks_dir_path = '/Users/kamildre/phd/osil/assets/tasks'
    else:
        tasks_dir_path = '/mnt/data/kamil/tasks'

    # get_mean_distance_between_waypoints_in_all_demons(tasks_dir_path)

    # task_names = [d for d in os.listdir(tasks_dir_path) if os.path.isdir(join(tasks_dir_path, d)) and d != 'processed']
    #
    # task_name = task_names[0]
    # traj_eef_posevec = np.load(join(tasks_dir_path, task_name, 'demo_eef_posevecs.npy'))
    # indices = get_subsample_indices(traj_eef_posevec, 0.01)

    dataset = BCDataset(tasks_dir_path, mt_act_args_dict, reprocess=True, visualise=True, add_noise_to_act_hist=True,
                        mask_pcd_clusters=True, processed_dir='bn_reaching_processed')

    for i in dataset:
        pass

    # dataset.load_demo_data(task_name)
