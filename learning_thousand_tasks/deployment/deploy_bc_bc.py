"""
BC-BC (BC Alignment + BC Interaction) Deployment Script

Demonstrates the BC-BC pipeline:
1. Load test image and pre-computed segmentation from assets/inference_example/
2. Run BC alignment policy to predict trajectory to bottleneck pose
3. Track waypoints until alignment policy signals termination (reached bottleneck)
4. Run BC interaction policy to predict manipulation trajectory
5. Track waypoints with gripper control until interaction policy signals termination
6. Visualize both alignment and interaction trajectories

For actual robot deployment:
- Replace test image loading with live camera capture and XMem tracking
- Track predicted alignment waypoints with a controller until terminate_prob > 0.95
- Once at bottleneck, track interaction waypoints with gripper control
- Update gripper state based on gripper_action_prob (low = open, high = close)
- Continue interaction until terminate_prob > 0.95
"""

import numpy as np
import torch
import clip
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from thousand_tasks.core.globals import ASSETS_DIR
from thousand_tasks.core.utils.scene_state import SceneState
from thousand_tasks.training.act_bn_reaching.policy import ACTPolicy as AlignmentPolicy
from thousand_tasks.training.act_bn_reaching.policy import get_action as alignment_get_action
from thousand_tasks.training.act_bn_reaching.config import mt_act_args_dict as alignment_args_dict
from thousand_tasks.training.act_interaction.policy import ACTPolicy as InteractionPolicy
from thousand_tasks.training.act_interaction.policy import get_action as interaction_get_action
from thousand_tasks.training.act_interaction.config import mt_act_args_dict as interaction_args_dict
from thousand_tasks.training.point_cloud_utils import downsample_pcd


def visualize_test_scene(rgb, depth, segmap, save_path):
    """Visualize test scene with RGB, depth, and segmentation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')

    axes[1].imshow(depth, cmap='viridis')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')

    segmented_rgb = rgb * segmap[..., None]
    axes[2].imshow(segmented_rgb)
    axes[2].set_title('Segmented RGB')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {save_path}")
    plt.close()


def visualize_alignment_trajectory(pcd_world, pcd_rgb, waypoints, T_WE_start, k_visualize=15):
    """Visualize predicted alignment trajectory in Open3D."""
    import open3d as o3d

    # Create point cloud
    o3d_pcd_points = o3d.geometry.PointCloud()
    o3d_pcd_points.points = o3d.utility.Vector3dVector(pcd_world)
    o3d_pcd_points.colors = o3d.utility.Vector3dVector(pcd_rgb)

    geometries = [o3d_pcd_points]

    # Add starting pose (green)
    coord_frame_start = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.08, origin=[0, 0, 0])
    coord_frame_start.transform(T_WE_start)
    geometries.append(coord_frame_start)

    # Add waypoints with fading
    num_waypoints = min(len(waypoints), k_visualize)
    for i in range(num_waypoints):
        opacity = 1.0 - (i / num_waypoints) * 0.8
        size = 0.06 * opacity
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size, origin=[0, 0, 0])
        coord_frame.transform(waypoints[i])
        geometries.append(coord_frame)

    # Add trajectory line
    if len(waypoints) > 1:
        trajectory_points = [T_WE_start[:3, 3]] + [wp[:3, 3] for wp in waypoints[:k_visualize]]
        lines = [[i, i+1] for i in range(len(trajectory_points)-1)]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(trajectory_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in lines])  # Green
        geometries.append(line_set)

    print(f"  Displaying ALIGNMENT trajectory (showing {num_waypoints} waypoints)...")
    print("  Close window to continue...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="BC-BC: Alignment Phase Trajectory",
        width=1024,
        height=768,
        point_show_normal=False
    )


def visualize_interaction_trajectory(pcd_world, pcd_rgb, waypoints, T_WE_bottleneck, k_visualize=10):
    """Visualize predicted interaction trajectory in Open3D."""
    import open3d as o3d

    # Create point cloud
    o3d_pcd_points = o3d.geometry.PointCloud()
    o3d_pcd_points.points = o3d.utility.Vector3dVector(pcd_world)
    o3d_pcd_points.colors = o3d.utility.Vector3dVector(pcd_rgb)

    geometries = [o3d_pcd_points]

    # Add bottleneck pose (red, larger)
    coord_frame_bottleneck = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.10, origin=[0, 0, 0])
    coord_frame_bottleneck.transform(T_WE_bottleneck)
    geometries.append(coord_frame_bottleneck)

    # Add waypoints with fading
    num_waypoints = min(len(waypoints), k_visualize)
    for i in range(num_waypoints):
        opacity = 1.0 - (i / num_waypoints) * 0.8
        size = 0.06 * opacity
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size, origin=[0, 0, 0])
        coord_frame.transform(waypoints[i])
        geometries.append(coord_frame)

    # Add trajectory line
    if len(waypoints) > 1:
        trajectory_points = [T_WE_bottleneck[:3, 3]] + [wp[:3, 3] for wp in waypoints[:k_visualize]]
        lines = [[i, i+1] for i in range(len(trajectory_points)-1)]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(trajectory_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])  # Red
        geometries.append(line_set)

    print(f"  Displaying INTERACTION trajectory (showing {num_waypoints} waypoints)...")
    print("  Close window to continue...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="BC-BC: Interaction Phase Trajectory",
        width=1024,
        height=768,
        point_show_normal=False
    )


def main():
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    inference_dir = ASSETS_DIR / 'inference_example'
    checkpoint_dir_alignment = ASSETS_DIR / 'checkpoints' / 'bc_alignment'
    checkpoint_dir_interaction = ASSETS_DIR / 'checkpoints' / 'bc_interaction'

    # Task-specific parameters
    task_name = 'pick_up_shoe'

    # Policy parameters
    alignment_checkpoint_name = 'best.pt'
    interaction_checkpoint_name = 'best.pt'
    k_visualize_alignment = 15  # Number of alignment waypoints to visualize
    k_visualize_interaction = 10  # Number of interaction waypoints to visualize
    k_track = 3  # Number of waypoints to track per inference cycle (open-loop horizon)

    # Simulated starting pose (in practice, read from robot)
    T_WE_start = np.array([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.6],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Simulated bottleneck pose (reached after alignment phase)
    # In practice, this would be the final pose from alignment policy
    T_WE_bottleneck = np.array([
        [1.0, 0.0, 0.0, 0.55],
        [0.0, 1.0, 0.0, 0.05],
        [0.0, 0.0, 1.0, 0.45],
        [0.0, 0.0, 0.0, 1.0]
    ])

    print("="*80)
    print("BC-BC Deployment (BC Alignment + BC Interaction)")
    print("="*80)

    # -------------------------------------------------------------------------
    # Step 1: Load test image
    # -------------------------------------------------------------------------
    print("\nStep 1: Load test image")

    test_rgb = np.array(Image.open(str(inference_dir / 'head_camera_ws_rgb.png')))
    test_depth = np.array(Image.open(str(inference_dir / 'head_camera_ws_depth_to_rgb.png')))
    test_segmap = np.load(str(inference_dir / 'head_camera_ws_segmap.npy'))
    intrinsics = np.load(str(inference_dir / 'head_camera_rgb_intrinsic_matrix.npy'))
    T_WC = np.load(str(ASSETS_DIR / 'T_WC_head.npy'))

    print(f"  Loaded RGB: {test_rgb.shape}, Depth: {test_depth.shape}, Segmap: {test_segmap.shape}")

    # Visualize test scene
    vis_dir = ASSETS_DIR / 'example_visualisations'
    vis_dir.mkdir(exist_ok=True)
    save_path = vis_dir / 'bc_bc_test_scene.png'
    visualize_test_scene(test_rgb, test_depth, test_segmap, save_path)

    # -------------------------------------------------------------------------
    # Step 2: Initialize live scene state
    # -------------------------------------------------------------------------
    print("\nStep 2: Initialize live scene state")
    live_scene_state = SceneState.initialise_from_dict({
        'rgb': test_rgb,
        'depth': test_depth,
        'segmap': test_segmap,
        'intrinsic_matrix': intrinsics
    })

    live_scene_state.erode_segmap()
    live_scene_state.crop_object_using_segmap()
    print(f"  Processed segmentation for: '{task_name}'")

    # Visualize point cloud
    import open3d as o3d
    pcd = live_scene_state.o3d_pcd
    print(f"  Point cloud has {len(pcd.points)} points")

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    print("  Displaying point cloud (close window to continue)...")
    o3d.visualization.draw_geometries(
        [pcd, coord_frame],
        window_name="Live Scene Point Cloud",
        width=800,
        height=600,
        point_show_normal=False
    )

    # =========================================================================
    # PART 1: ALIGNMENT PHASE - BC policy predicts trajectory to bottleneck
    # =========================================================================

    print("\n\nPART 1: ALIGNMENT PHASE - BC POLICY")
    print("="*80)

    # -------------------------------------------------------------------------
    # Step 3: Load alignment policy
    # -------------------------------------------------------------------------
    print("\nStep 3: Load trained alignment policy")

    alignment_checkpoint_path = checkpoint_dir_alignment / alignment_checkpoint_name
    if not alignment_checkpoint_path.exists():
        print(f"\n  WARNING: Alignment checkpoint not found at {alignment_checkpoint_path}")
        print("  This is expected if you haven't trained the alignment policy yet.")
        print("  Skipping policy inference...")
        print("\n  To train the alignment policy:")
        print("    1. Preprocess demonstrations: make preprocess_demos")
        print("    2. Create training dataset: make create_alignment_dataset")
        print("    3. Train policy: make train_bc_alignment")
        return

    # Initialize alignment policy
    alignment_policy = AlignmentPolicy(alignment_args_dict)
    alignment_policy.cuda() if torch.cuda.is_available() else alignment_policy.cpu()

    # Load checkpoint
    checkpoint = torch.load(str(alignment_checkpoint_path), map_location='cuda' if torch.cuda.is_available() else 'cpu')
    alignment_policy.load_state_dict(checkpoint)
    alignment_policy.eval()

    print(f"  Loaded alignment policy from: {alignment_checkpoint_path}")

    # -------------------------------------------------------------------------
    # Step 4: Initialize CLIP for language conditioning
    # -------------------------------------------------------------------------
    print("\nStep 4: Initialize language features")

    clip_model, _ = clip.load("ViT-B/32", device='cpu')
    text_tokens = clip.tokenize([task_name.replace('_', ' ')])
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()

    print(f"  Encoded task: '{task_name.replace('_', ' ')}'")

    # -------------------------------------------------------------------------
    # Step 5: Run alignment policy inference
    # -------------------------------------------------------------------------
    print("\nStep 5: Run alignment policy to predict trajectory to bottleneck")

    # Prepare point cloud
    o3d_pcd_world = live_scene_state.o3d_pcd
    pcd_world = np.asarray(o3d_pcd_world.points)
    pcd_rgb = np.asarray(o3d_pcd_world.colors)
    pcd_world, pcd_rgb = downsample_pcd(pcd_world, pcd_rgb, 2048)
    o3d_pcd_world.points = o3d.utility.Vector3dVector(pcd_world)
    o3d_pcd_world.colors = o3d.utility.Vector3dVector(pcd_rgb)

    # Initialize action history for alignment
    past_actions_alignment = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * alignment_args_dict['history_size']
    )
    gripper_state = 0  # Closed

    # Run alignment inference
    with torch.no_grad():
        alignment_waypoints, _, alignment_terminate_prob, _ = alignment_get_action(
            policy=alignment_policy,
            o3d_pcd_world=o3d_pcd_world,
            T_WE=T_WE_start,
            gripper_state=gripper_state,
            text_features=text_features,
            past_actions=past_actions_alignment,
            plotter=None,
            shown=False
        )

    print(f"  Alignment policy predicted {len(alignment_waypoints)} waypoints")
    print(f"  First waypoint relative position: {(alignment_waypoints[0] @ np.linalg.inv(T_WE_start))[:3, 3]}")

    # -------------------------------------------------------------------------
    # Step 6: Visualize alignment trajectory
    # -------------------------------------------------------------------------
    print("\nStep 6: Visualize alignment trajectory")

    visualize_alignment_trajectory(
        pcd_world=pcd_world,
        pcd_rgb=pcd_rgb,
        waypoints=alignment_waypoints,
        T_WE_start=T_WE_start,
        k_visualize=k_visualize_alignment
    )

    print("\n" + "="*80)
    print("ALIGNMENT PHASE INSTRUCTIONS")
    print("="*80)
    print(f"Track waypoints with a controller until alignment policy signals termination:")
    print(f"  1. Use a Cartesian controller to track {k_track} waypoints per cycle")
    print(f"  2. After each cycle, run alignment policy again with:")
    print(f"     - Updated point cloud observation")
    print(f"     - Current end-effector pose")
    print(f"     - Updated action history")
    print(f"  3. Continue until terminate_prob > 0.95 (bottleneck reached)")
    print("="*80)

    print(f"\nFirst {k_track} alignment waypoints:")
    for i in range(min(k_track, len(alignment_waypoints))):
        print(f"  Waypoint {i+1}:")
        print(f"    Position: {alignment_waypoints[i][:3, 3]}")
        print(f"    Terminate prob: {alignment_terminate_prob[i]:.3f}")

    # =========================================================================
    # PART 2: INTERACTION PHASE - BC policy predicts manipulation trajectory
    # =========================================================================

    print("\n\nPART 2: INTERACTION PHASE - BC POLICY")
    print("="*80)

    # -------------------------------------------------------------------------
    # Step 7: Load interaction policy
    # -------------------------------------------------------------------------
    print("\nStep 7: Load trained interaction policy")

    interaction_checkpoint_path = checkpoint_dir_interaction / interaction_checkpoint_name
    if not interaction_checkpoint_path.exists():
        print(f"\n  WARNING: Interaction checkpoint not found at {interaction_checkpoint_path}")
        print("  This is expected if you haven't trained the interaction policy yet.")
        print("  Skipping interaction inference...")
        print("\n  To train the interaction policy:")
        print("    1. Preprocess demonstrations: make preprocess_demos")
        print("    2. Create training dataset: make create_interaction_dataset")
        print("    3. Train policy: make train_bc_interaction")
        return

    # Initialize interaction policy
    interaction_policy = InteractionPolicy(interaction_args_dict)
    interaction_policy.cuda() if torch.cuda.is_available() else interaction_policy.cpu()

    # Load checkpoint
    checkpoint = torch.load(str(interaction_checkpoint_path), map_location='cuda' if torch.cuda.is_available() else 'cpu')
    interaction_policy.load_state_dict(checkpoint)
    interaction_policy.eval()

    print(f"  Loaded interaction policy from: {interaction_checkpoint_path}")

    # -------------------------------------------------------------------------
    # Step 8: Simulate reaching bottleneck pose
    # -------------------------------------------------------------------------
    print("\nStep 8: [SIMULATED] Assume robot has reached bottleneck pose")
    print("  In practice: Continue tracking alignment waypoints until terminate_prob > 0.95")
    print(f"  Bottleneck position: {T_WE_bottleneck[:3, 3]}")

    # -------------------------------------------------------------------------
    # Step 9: Run interaction policy inference
    # -------------------------------------------------------------------------
    print("\nStep 9: Run interaction policy to predict manipulation trajectory")

    # Initialize action history for interaction (gripper starts closed)
    past_actions_interaction = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]] * interaction_args_dict['history_size']
    )  # [vx, vy, vz, wx, wy, wz, gripper, terminate]

    # Run interaction inference
    with torch.no_grad():
        interaction_waypoints, gripper_action_prob, interaction_terminate_prob, _ = interaction_get_action(
            policy=interaction_policy,
            o3d_pcd_world=o3d_pcd_world,
            T_WE=T_WE_bottleneck,
            gripper_state=gripper_state,
            text_features=text_features,
            past_actions=past_actions_interaction,
            plotter=None,
            shown=False
        )

    print(f"  Interaction policy predicted {len(interaction_waypoints)} waypoints")
    print(f"  First waypoint relative position: {(interaction_waypoints[0] @ np.linalg.inv(T_WE_bottleneck))[:3, 3]}")

    # -------------------------------------------------------------------------
    # Step 10: Visualize interaction trajectory
    # -------------------------------------------------------------------------
    print("\nStep 10: Visualize interaction trajectory")

    visualize_interaction_trajectory(
        pcd_world=pcd_world,
        pcd_rgb=pcd_rgb,
        waypoints=interaction_waypoints,
        T_WE_bottleneck=T_WE_bottleneck,
        k_visualize=k_visualize_interaction
    )

    # -------------------------------------------------------------------------
    # Final instructions
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("INTERACTION PHASE INSTRUCTIONS")
    print("="*80)
    print(f"Track waypoints with gripper control until interaction policy signals termination:")
    print(f"  1. Use a Cartesian controller to track {k_track} waypoints per cycle")
    print(f"  2. Update gripper state based on gripper_action_prob (gripper closure probability):")
    print(f"     - Open gripper if prob < 0.1 (low closure = should be open)")
    print(f"     - Close gripper if prob > 0.9 (high closure = should be closed)")
    print(f"  3. After each cycle, run interaction policy again with:")
    print(f"     - Updated point cloud observation")
    print(f"     - Current end-effector pose")
    print(f"     - Updated action history")
    print(f"  4. Continue until terminate_prob > 0.895 (task complete)")
    print("="*80)

    print(f"\nFirst {k_track} interaction waypoints:")
    for i in range(min(k_track, len(interaction_waypoints))):
        print(f"  Waypoint {i+1}:")
        print(f"    Position: {interaction_waypoints[i][:3, 3]}")
        print(f"    Gripper prob: {gripper_action_prob[i]:.3f}")
        print(f"    Terminate prob: {interaction_terminate_prob[i]:.3f}")

    print("\n" + "="*80)
    print("DEPLOYMENT COMPLETE")
    print("="*80)
    print("BC-BC method uses two learned policies:")
    print("  1. Alignment policy: Predicts trajectory to bottleneck (until terminate > 0.95)")
    print("  2. Interaction policy: Predicts manipulation trajectory (until terminate > 0.95)")
    print("\nBoth policies run in closed-loop with re-inference after every few waypoints.")
    print("="*80)


if __name__ == '__main__':
    main()
