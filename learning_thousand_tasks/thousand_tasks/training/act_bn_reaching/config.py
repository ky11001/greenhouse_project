import numpy as np


class MTACTARGS:
    def __init__(self):
        for key in mt_act_args_dict:
            setattr(self, key, mt_act_args_dict[key])


hidden_dim = 256

mt_act_args_dict = {
    ###############################################################
    # MT-ACT parameters.
    ###############################################################
    'action_mode': 'abs_delta',  # 'abs_delta' or 'delta'
    'open_loop_horizon': 3,  # Number of steps to predict in open loop.
    'state_dim': 7,  # 3 trans, 3 angle axis, 1 gripper (optional 1 terminal)  # TODO key difference
    'position_embedding': 'sine',
    'enc_layers': 4,
    'dec_layers': 7,
    'dim_feedforward': 2048,
    'hidden_dim': hidden_dim,  # If you change this, you need to change pointnet encoder to match.
    'dropout': 0.1,
    'nheads': 8,
    'chunk_size': 6,  # 3 actions ~ 1s
    'history_size': 12,
    'pre_norm': False,
    'use_film': True,
    'lang_dim': 512,
    ###############################################################
    # Training parameters.
    ###############################################################
    'seed': 42,
    'device': 'cuda',
    'num_epochs': 10000000000000,
    'kl_weight': 10,
    'lr': 0.0001,
    'batch_size': 16,
    'weight_decay': 0.0001,
    'epochs': 100000000000,
    'lr_drop': 200,
    'clip_max_norm': 0.1,
    ###############################################################
    # Parameters for data augmentation.
    ###############################################################
    'num_traj_to_bn_train': 10,  # Number of synthetic trajectories to generate per demonstration
    'corrective_action_ratio': 0.5,
    # 'use_bn_reaching': False,  # If False, ignores the actions to reach the bottleneck.
    # 'sub_sampling_factor': 20,  # Leaves every nth sample in the trajectory when processing the data.
    'max_num_points': 2048,  # Number of points used for inference
    'num_points_to_save': 4096,  # Number of points to save
    'preprocessing_num_clusters_total': 10,
    'preprocessing_num_clusters_to_mask': 4,
    'preprocessing_pcd_noise_std': 1.5e-3,
    'preprocessing_act_hist_noise_std': 0.4,  # 0.1 looks ok when visualising
    'trans_spacing_between_waypoints': 1e-2,
    'trans_perturb_to_bn_lb': np.array([-0.4, -0.4, -0.1]),
    'trans_perturb_to_bn_ub': np.array([0.4, 0.4, 0.2]),
    'rot_perturb_to_bn_lb': np.array([-np.deg2rad(-0), -np.deg2rad(-0), -np.deg2rad(45)]),
    'rot_perturb_to_bn_ub': np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(45)]),

    # 'trans_perturb_demo_lb': np.array([-0.02, -0.02, -0.02]),
    # 'trans_perturb_demo_ub': np.array([0.02, 0.02, 0.02]),
    # 'rot_perturb_demo_lb': np.array([-np.deg2rad(10), -np.deg2rad(10), -np.deg2rad(10)]),
    # 'rot_perturb_demo_ub': np.array([np.deg2rad(10), np.deg2rad(10), np.deg2rad(10)]),
    ###############################################################
    # PointNet encoder parameters.
    ###############################################################
    'ratio_1': 0.1,
    'radius_1': 0.1,
    'ratio_2': 0.1,
    'radius_2': 0.2,
    'locla_nn_dims_1': [3 + 3, 64, 64, 128],
    'global_nn_dims_1': [128, 128, 128],
    'locla_nn_dims_2': [128 + 3, 128, 128, 256],
    # 'global_nn_dims_2': [256, 256, 256],   # last one needs to match the hidden_dim
    'global_nn_dims_2': [256, 256, hidden_dim],  # last one needs to match the hidden_dim    # TODO this was changed
    'num_freqs_pt': 10,
    'num_freqs_pe': 10,
    'cat_pos': True,

    ###############################################################
    # Legacy parameters.
    ###############################################################
    'backbone': 'resnet18',
    'lr_backbone': 1e-4,
    'camera_names': None,
}

mt_act_args = MTACTARGS()
