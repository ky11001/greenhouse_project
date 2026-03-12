import argparse
import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from thousand_tasks.training.act_interaction.dataset import BCDataset
from thousand_tasks.training.act_interaction.config import mt_act_args_dict
from thousand_tasks.training.act_interaction.policy import ACTPolicy
from thousand_tasks.training.act_interaction.evaluator import Evaluator
from thousand_tasks.core.globals import TASKS_DIR, CHECKPOINTS_DIR

if __name__ == '__main__':
    # --- Arg parser ---
    parser = argparse.ArgumentParser(description="BC Interaction Training Script")
    parser.add_argument('-name', '--name', type=str, required=False,
                        help='Name for the training run (used for checkpoint naming)', default='bc_interaction')
    parser.add_argument('-cam', '--camera', type=str, required=False,
                        help="Camera to use: 'external' or 'head'", default='head')
    parser.add_argument('-device', '--device', type=str, required=False,
                        help='Device: cpu or cuda', default='cuda')
    parser.add_argument('-batch_size', '--batch_size', type=int, required=False,
                        help='Batch size (overrides config)', default=None)
    parser.add_argument('-hd', '--hidden_dimension', type=int, required=False,
                        help='Hidden dimension in transformer (overrides config)', default=None)
    parser.add_argument('-num_workers', '--num_workers', type=int, required=False,
                        help='Number of data loader workers', default=0)
    parser.add_argument('-max_epochs', '--max_epochs', type=int, required=False,
                        help='Maximum number of training epochs', default=1000)
    args = parser.parse_args()
    # --- End ---

    # Setup paths
    root = str(TASKS_DIR)
    checkpoint_dir = CHECKPOINTS_DIR / args.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Update config with command line arguments
    batch_size = mt_act_args_dict['batch_size'] if args.batch_size is None else args.batch_size
    if args.hidden_dimension is not None:
        mt_act_args_dict['hidden_dim'] = args.hidden_dimension
        mt_act_args_dict['nheads'] = int(args.hidden_dimension / 64)

    folder_name = 'interaction_processed'

    dset_train = BCDataset(root=root,
                           config=mt_act_args_dict,
                           mask_pcd_clusters=True,
                           add_noise_to_pcd=True,
                           add_noise_to_act_hist=True,
                           perturb_interaction_traj=True,
                           visualise=False,
                           processed_dir=folder_name,
                           which_camera=args.camera)

    print(f'Number of data points (train): {len(dset_train)}')

    # Regular DataLoader with shuffle (no custom sampler needed for interaction training)
    data_loader_train = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    mt_act_args_dict['device'] = args.device
    policy = ACTPolicy(mt_act_args_dict)
    policy.trans_rot_min = dset_train.trans_rot_min
    policy.trans_rot_max = dset_train.trans_rot_max
    policy.trans_rot_min_hist = dset_train.trans_rot_min_hist
    policy.trans_rot_max_hist = dset_train.trans_rot_max_hist
    policy.num_term_acts = torch.tensor(dset_train.num_term_acts)
    policy.num_dont_term_acts = torch.tensor(dset_train.num_not_term_acts)
    policy.to(args.device)

    optimizer = policy.configure_optimizers()

    evaluator = Evaluator(root=root, config=mt_act_args_dict, which_camera=args.camera)

    num_epochs = args.max_epochs

    best_trans_error = np.inf
    best_rot_error = np.inf
    best_sum_error = np.inf

    total_loss_trans_l1_train = []
    total_loss_rot_l1_train = []
    total_loss_kl_train = []
    total_loss_gripper_bce_train = []
    total_rot_error_train = []
    total_loss_termination_bce_train = []
    total_loss_train = []
    total_trans_error_train = []

    try:
        for epoch in range(num_epochs):
            # Training
            policy.train()
            optimizer.zero_grad()
            for i, data in tqdm(enumerate(data_loader_train), desc=f'(Training)  Epoch {epoch}/{num_epochs}',
                                total=len(data_loader_train), leave=False):
                forward_dict = policy(data.to(args.device))
                loss = forward_dict['loss']
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Enforce constraints on loss weights
                with torch.no_grad():
                    policy.var_trans.clamp_(1e-5)
                    policy.var_rot.clamp_(1e-5)
                    policy.var_gripper.clamp_(1e-5)
                    policy.var_termination.clamp_(1e-5)

                batch_size = len(data.text_features)
                total_loss_trans_l1_train.append(forward_dict['Trans_L1'].item() * batch_size)
                total_loss_rot_l1_train.append(forward_dict['Rot_L1'].item() * batch_size)
                total_loss_kl_train.append(forward_dict['KL'].item() * batch_size)
                total_loss_gripper_bce_train.append(forward_dict['Gripper_BCE'].item() * batch_size)
                total_loss_termination_bce_train.append(forward_dict['Termination_BCE'].item() * batch_size)
                total_loss_train.append(forward_dict['loss'].item() * batch_size)
                total_trans_error_train.append(forward_dict['Trans_Error'] * batch_size)
                total_rot_error_train.append(forward_dict['Rot_Error'] * batch_size)

            # Evaluate
            policy.eval()
            with torch.no_grad():
                pos_error, rot_error = evaluator.evaluate_policy(policy, num_rollouts_per_task=50)

            # Compute average metrics
            avg_trans_loss = np.sum(total_loss_trans_l1_train) / len(dset_train)
            avg_trans_error_mm = np.sum(total_trans_error_train) * 1e3 / len(dset_train)
            avg_rot_error_deg = np.sum(total_rot_error_train) / len(dset_train)
            avg_total_loss = np.sum(total_loss_train) / len(dset_train)

            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'  Train - Loss: {avg_total_loss:.4f}, Trans Loss: {avg_trans_loss:.4f}, Trans Error: {avg_trans_error_mm:.2f} mm, Rot Error: {avg_rot_error_deg:.2f} deg')
            print(f'  Eval  - Position Error: {pos_error:.2f} mm, Rotation Error: {rot_error:.2f} deg')

            # Save best models (only when we improve)
            if pos_error + rot_error < best_sum_error:
                best_sum_error = pos_error + rot_error
                best_trans_error = pos_error
                best_rot_error = rot_error

                # Save the best checkpoint
                checkpoint_path = checkpoint_dir / 'best.pt'
                torch.save(policy.state_dict(), str(checkpoint_path))

                # Save metrics
                metrics_path = checkpoint_dir / 'best_metrics.txt'
                with open(str(metrics_path), 'w') as f:
                    f.write(f'Epoch: {epoch + 1}\n')
                    f.write(f'Position Error: {pos_error:.2f} mm\n')
                    f.write(f'Rotation Error: {rot_error:.2f} deg\n')
                    f.write(f'Combined Error: {best_sum_error:.2f}\n')

                print(f'  *** New best model saved! (combined error: {best_sum_error:.2f}) ***')

            total_loss_trans_l1_train = []
            total_loss_rot_l1_train = []
            total_loss_kl_train = []
            total_loss_gripper_bce_train = []
            total_rot_error_train = []
            total_loss_termination_bce_train = []
            total_loss_train = []
            total_trans_error_train = []

    except KeyboardInterrupt:
        print('\nTraining interrupted by user')

    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / 'final.pt'
    torch.save(policy.state_dict(), str(final_checkpoint_path))
    print(f'\nFinal checkpoint saved to: {final_checkpoint_path}')
    print(f'Best checkpoint saved to: {checkpoint_dir / "best.pt"}')
