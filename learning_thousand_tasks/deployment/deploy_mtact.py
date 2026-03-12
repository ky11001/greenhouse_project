"""
MT-ACT+ (End-to-End Multi-Task Transformer) Deployment Script

Demonstrates the MT-ACT+ pipeline:
1. Load test image and pre-computed segmentation from assets/inference_example/
2. Run end-to-end MT-ACT+ policy to predict complete manipulation trajectory
3. Track waypoints with gripper control until policy signals termination
4. Visualize predicted trajectory

MT-ACT+ is an end-to-end baseline that directly predicts action sequences
without explicit decomposition into alignment and interaction phases.

For actual robot deployment:
- Replace test image loading with live camera capture and XMem tracking
- Track predicted waypoints with a controller
- Update gripper state based on gripper_action_prob (low = open, high = close)
- Re-run inference after every k waypoints for closed-loop control
- Continue until terminate_prob > 0.95
"""

import numpy as np
import torch
import clip
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from thousand_tasks.core.globals import ASSETS_DIR
from thousand_tasks.core.utils.scene_state import SceneState
from thousand_tasks.training.act_interaction.policy import ACTPolicy, get_action
from thousand_tasks.training.act_end_to_end.config import mt_act_args_dict
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


def visualize_predicted_trajectory(pcd_world, pcd_rgb, waypoints, T_WE_start, k_visualize=10):
    """
    Visualize predicted end-to-end trajectory in Open3D.

    Args:
        pcd_world: Object point cloud in world frame (N x 3)
        pcd_rgb: RGB colors for point cloud (N x 3, range 0-1)
        waypoints: List of predicted SE(3) poses (4x4 matrices)
        T_WE_start: Starting end-effector pose (4x4 matrix)
        k_visualize: Number of future waypoints to visualize
    """
    import open3d as o3d

    # Create point cloud
    o3d_pcd_points = o3d.geometry.PointCloud()
    o3d_pcd_points.points = o3d.utility.Vector3dVector(pcd_world)
    o3d_pcd_points.colors = o3d.utility.Vector3dVector(pcd_rgb)

    geometries = [o3d_pcd_points]

    # Add starting pose (bright green)
    coord_frame_start = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.08, origin=[0, 0, 0])
    coord_frame_start.transform(T_WE_start)
    geometries.append(coord_frame_start)

    # Add waypoints with fading opacity
    num_waypoints = min(len(waypoints), k_visualize)
    for i in range(num_waypoints):
        # Calculate opacity (fade from 1.0 to 0.2)
        opacity = 1.0 - (i / num_waypoints) * 0.8

        # Create coordinate frame
        size = 0.06 * opacity  # Also reduce size for far waypoints
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size, origin=[0, 0, 0])
        coord_frame.transform(waypoints[i])

        geometries.append(coord_frame)

    # Add trajectory line connecting waypoints
    if len(waypoints) > 1:
        trajectory_points = [T_WE_start[:3, 3]] + [wp[:3, 3] for wp in waypoints[:k_visualize]]
        lines = [[i, i+1] for i in range(len(trajectory_points)-1)]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(trajectory_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in lines])  # Blue trajectory
        geometries.append(line_set)

    # Visualize
    print(f"  Displaying predicted trajectory (showing {num_waypoints} waypoints)...")
    print("  Close window to continue...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="MT-ACT+: End-to-End Predicted Trajectory",
        width=1024,
        height=768,
        point_show_normal=False
    )


def main():
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    inference_dir = ASSETS_DIR / 'inference_example'
    checkpoint_dir = ASSETS_DIR / 'checkpoints' / 'mtact_plus'

    # Task-specific parameters
    task_name = 'pick_up_shoe'

    # Policy parameters
    checkpoint_name = 'best.pt'  # Update with your checkpoint name
    k_visualize = 15  # Number of waypoints to visualize
    k_track = 3  # Number of waypoints to track per inference cycle (open-loop horizon)

    # Simulated starting pose (in practice, read from robot)
    T_WE_start = np.array([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.6],
        [0.0, 0.0, 0.0, 1.0]
    ])

    print("="*80)
    print("MT-ACT+ Deployment (End-to-End Multi-Task Transformer)")
    print("="*80)

    # -------------------------------------------------------------------------
    # Step 1: Load test image
    # -------------------------------------------------------------------------
    print("\nStep 1: Load test image")

    # Load RGB and depth from workspace images
    test_rgb = np.array(Image.open(str(inference_dir / 'head_camera_ws_rgb.png')))
    test_depth = np.array(Image.open(str(inference_dir / 'head_camera_ws_depth_to_rgb.png')))

    # Load segmentation mask for the target object (pre-computed)
    test_segmap = np.load(str(inference_dir / 'head_camera_ws_segmap.npy'))

    # Load camera intrinsics
    intrinsics = np.load(str(inference_dir / 'head_camera_rgb_intrinsic_matrix.npy'))

    # Load camera extrinsics (world-to-camera transform)
    T_WC = np.load(str(ASSETS_DIR / 'T_WC_head.npy'))

    print(f"  Loaded RGB: {test_rgb.shape}, Depth: {test_depth.shape}, Segmap: {test_segmap.shape}")

    # Visualize test image, depth, and segmentation
    vis_dir = ASSETS_DIR / 'example_visualisations'
    vis_dir.mkdir(exist_ok=True)

    save_path = vis_dir / 'mtact_test_scene.png'
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

    # Process segmentation: erode mask and crop to object region
    live_scene_state.erode_segmap()
    live_scene_state.crop_object_using_segmap()
    print(f"  Processed segmentation for: '{task_name}'")

    # Visualize point cloud with Open3D
    import open3d as o3d
    pcd = live_scene_state.o3d_pcd
    print(f"  Point cloud has {len(pcd.points)} points")

    # Create coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    print("  Displaying point cloud (close window to continue)...")
    o3d.visualization.draw_geometries(
        [pcd, coord_frame],
        window_name="Live Scene Point Cloud",
        width=800,
        height=600,
        point_show_normal=False
    )

    # -------------------------------------------------------------------------
    # Step 3: Load MT-ACT+ policy
    # -------------------------------------------------------------------------
    print("\nStep 3: Load trained MT-ACT+ policy")

    checkpoint_path = checkpoint_dir / checkpoint_name
    if not checkpoint_path.exists():
        print(f"\n  WARNING: Checkpoint not found at {checkpoint_path}")
        print("  This is expected if you haven't trained the MT-ACT+ policy yet.")
        print("  Skipping policy inference...")
        print("\n  To train the MT-ACT+ policy:")
        print("    1. Preprocess demonstrations: make preprocess_demos")
        print("    2. Create training dataset: make create_mtact_dataset")
        print("    3. Train policy: make train_mtact")
        print("    4. Copy checkpoint to assets/checkpoints/mtact_plus/")
        return

    # Initialize policy
    policy = ACTPolicy(mt_act_args_dict)
    policy.cuda() if torch.cuda.is_available() else policy.cpu()

    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location='cuda' if torch.cuda.is_available() else 'cpu')
    policy.load_state_dict(checkpoint)
    policy.eval()

    print(f"  Loaded policy from: {checkpoint_path}")

    # -------------------------------------------------------------------------
    # Step 4: Initialize CLIP for language conditioning
    # -------------------------------------------------------------------------
    print("\nStep 4: Initialize language features")

    # Initialize CLIP for text encoding
    clip_model, _ = clip.load("ViT-B/32", device='cpu')

    # Encode task name
    text_tokens = clip.tokenize([task_name.replace('_', ' ')])
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()

    print(f"  Encoded task: '{task_name.replace('_', ' ')}'")

    # -------------------------------------------------------------------------
    # Step 5: Prepare input for policy
    # -------------------------------------------------------------------------
    print("\nStep 5: Prepare input for MT-ACT+ policy")

    # Get object point cloud in world frame
    o3d_pcd_world = live_scene_state.o3d_pcd
    pcd_world = np.asarray(o3d_pcd_world.points)
    pcd_rgb = np.asarray(o3d_pcd_world.colors)

    # Downsample to match training
    pcd_world, pcd_rgb = downsample_pcd(pcd_world, pcd_rgb, 2048)

    # Update Open3D point cloud
    o3d_pcd_world.points = o3d.utility.Vector3dVector(pcd_world)
    o3d_pcd_world.colors = o3d.utility.Vector3dVector(pcd_rgb)

    print(f"  Point cloud has {len(pcd_world)} points after downsampling")

    # Initialize action history (for temporal context)
    # MT-ACT+ format: [vx, vy, vz, wx, wy, wz, gripper, terminate]
    past_actions = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]] * mt_act_args_dict['history_size']
    )

    # Simulated current robot state
    gripper_state = 0  # Closed

    # -------------------------------------------------------------------------
    # Step 6: Run policy inference
    # -------------------------------------------------------------------------
    print("\nStep 6: Run MT-ACT+ policy inference to predict end-to-end trajectory")

    # Run inference
    with torch.no_grad():
        waypoints, gripper_action_prob, terminate_prob, _ = get_action(
            policy=policy,
            o3d_pcd_world=o3d_pcd_world,
            T_WE=T_WE_start,
            gripper_state=gripper_state,
            text_features=text_features,
            past_actions=past_actions,
            plotter=None,
            shown=False
        )

    print(f"  Policy predicted {len(waypoints)} waypoints")
    print(f"  First waypoint position (relative to start): {(waypoints[0] @ np.linalg.inv(T_WE_start))[:3, 3]}")

    # -------------------------------------------------------------------------
    # Step 7: Visualize predicted trajectory
    # -------------------------------------------------------------------------
    print("\nStep 7: Visualize predicted end-to-end trajectory")

    visualize_predicted_trajectory(
        pcd_world=pcd_world,
        pcd_rgb=pcd_rgb,
        waypoints=waypoints,
        T_WE_start=T_WE_start,
        k_visualize=k_visualize
    )

    # -------------------------------------------------------------------------
    # Final instructions
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("MT-ACT+ DEPLOYMENT INSTRUCTIONS")
    print("="*80)
    print(f"Track waypoints with gripper control until policy signals termination:")
    print(f"  1. Use a Cartesian controller to track {k_track} waypoints per cycle")
    print(f"  2. Update gripper state based on gripper_action_prob (gripper closure probability):")
    print(f"     - Open gripper if prob < 0.1 (low closure = should be open)")
    print(f"     - Close gripper if prob > 0.9 (high closure = should be closed)")
    print(f"  3. After each cycle, run MT-ACT+ policy again with:")
    print(f"     - Updated point cloud observation (using XMem for tracking)")
    print(f"     - Current end-effector pose")
    print(f"     - Updated action history")
    print(f"  4. Continue until terminate_prob > 0.95 (task complete)")
    print("="*80)

    # Print first few waypoint details
    print(f"\nFirst {k_track} waypoints to track:")
    for i in range(min(k_track, len(waypoints))):
        print(f"  Waypoint {i+1}:")
        print(f"    Position: {waypoints[i][:3, 3]}")
        print(f"    Gripper prob: {gripper_action_prob[i]:.3f}")
        print(f"    Terminate prob: {terminate_prob[i]:.3f}")

    print("\n" + "="*80)
    print("KEY DIFFERENCE: END-TO-END APPROACH")
    print("="*80)
    print("MT-ACT+ is an end-to-end baseline that:")
    print("  - Does NOT decompose task into alignment + interaction phases")
    print("  - Directly predicts full manipulation trajectory from observations")
    print("  - Runs in closed-loop with re-inference every k waypoints")
    print("  - Handles both reaching and manipulation in a single policy")
    print("="*80)


if __name__ == '__main__':
    main()
