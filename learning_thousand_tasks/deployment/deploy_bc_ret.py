"""
BC-Ret (BC Alignment + Retrieval Interaction) Deployment Script

Demonstrates the BC-Ret pipeline:
1. Load test image and pre-computed segmentation from assets/inference_example/
2. Retrieve similar demonstration via hierarchical retrieval
3. Run BC alignment policy to predict trajectory to bottleneck pose
4. Visualize predicted alignment trajectory
5. [SIMULATE] Assume robot reaches bottleneck pose by tracking waypoints
6. Replay demonstrated end-effector velocities (like MT3)

For actual robot deployment:
- Replace test image loading with live camera capture
- Provide pre-computed segmentation masks for target objects
- Track predicted alignment waypoints with a controller
- Once bottleneck pose is reached, replay end-effector velocities at recorded frequency
- Update gripper state according to demonstrated velocities
"""

import numpy as np
import torch
import clip
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from thousand_tasks.core.globals import ASSETS_DIR
from thousand_tasks.core.utils.scene_state import SceneState
from thousand_tasks.core.utils.se3_tools import pose_inv
from thousand_tasks.retrieval.hierarchical_retrieval import HierarchicalRetrieval
from thousand_tasks.training.act_bn_reaching.policy import ACTPolicy, get_action
from thousand_tasks.training.act_bn_reaching.config import mt_act_args_dict
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


def visualize_retrieval(test_rgb, test_segmap, demo_rgb, demo_segmap, save_path):
    """Visualize test and retrieved demo with segmentation masks in 2x2 layout."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # First row: Live scene
    axes[0, 0].imshow(test_rgb)
    axes[0, 0].set_title('Live Scene - RGB')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(test_rgb * test_segmap[..., None])
    axes[0, 1].set_title('Live Scene - Segmented')
    axes[0, 1].axis('off')

    # Second row: Retrieved demo
    axes[1, 0].imshow(demo_rgb)
    axes[1, 0].set_title('Retrieved Demo - RGB')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(demo_rgb * demo_segmap[..., None])
    axes[1, 1].set_title('Retrieved Demo - Segmented')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f"  Saved retrieval visualization to: {save_path}")
    plt.close()


def visualize_point_clouds(live_pcd, demo_pcd):
    """Visualize live and retrieved demonstration point clouds side by side."""
    import open3d as o3d

    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    print("  Displaying point clouds (close window to continue)...")
    o3d.visualization.draw_geometries(
        [live_pcd, demo_pcd, coord_frame],
        window_name="Object from Live Scene and Retrieved Demo",
        width=800,
        height=600,
        point_show_normal=False
    )


def visualize_predicted_trajectory(pcd_world, pcd_rgb, waypoints, T_WE_start, bottleneck_pose, k_visualize=10, save_path=None):
    """
    Visualize predicted alignment trajectory in Open3D with future steps faded out.

    Args:
        pcd_world: Object point cloud in world frame (N x 3)
        pcd_rgb: RGB colors for point cloud (N x 3, range 0-1)
        waypoints: List of predicted SE(3) poses (4x4 matrices)
        T_WE_start: Starting end-effector pose (4x4 matrix)
        bottleneck_pose: Target bottleneck pose (4x4 matrix) - goal position
        k_visualize: Number of future waypoints to visualize
        save_path: Optional path to save visualization screenshot
    """
    import open3d as o3d

    # Create point cloud
    o3d_pcd_points = o3d.geometry.PointCloud()
    o3d_pcd_points.points = o3d.utility.Vector3dVector(pcd_world)
    o3d_pcd_points.colors = o3d.utility.Vector3dVector(pcd_rgb)

    # Create visualizations list
    geometries = [o3d_pcd_points]

    # Add starting pose (bright green, full opacity)
    coord_frame_start = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.08, origin=[0, 0, 0])
    coord_frame_start.transform(T_WE_start)
    geometries.append(coord_frame_start)

    # Add bottleneck target pose (bright red, full opacity)
    coord_frame_target = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.10, origin=[0, 0, 0])
    coord_frame_target.transform(bottleneck_pose)
    # Make it red by painting the axes
    geometries.append(coord_frame_target)

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

        # Note: Open3D doesn't support transparency well in draw_geometries,
        # so we just use size to indicate distance
        geometries.append(coord_frame)

    # Add trajectory line connecting waypoints
    if len(waypoints) > 1:
        trajectory_points = [T_WE_start[:3, 3]] + [wp[:3, 3] for wp in waypoints[:k_visualize]] + [bottleneck_pose[:3, 3]]
        lines = [[i, i+1] for i in range(len(trajectory_points)-1)]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(trajectory_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in lines])  # Green trajectory
        geometries.append(line_set)

    # Visualize
    print(f"  Displaying predicted alignment trajectory (showing {num_waypoints} waypoints)...")
    print("  Green = Start, Red = Target Bottleneck, Blue = Waypoints")
    print("  Close window to continue...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="BC-Ret: Predicted Alignment Trajectory",
        width=1024,
        height=768,
        point_show_normal=False
    )


def main():
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    inference_dir = ASSETS_DIR / 'inference_example'
    demo_dir = ASSETS_DIR / 'demonstrations'
    checkpoint_dir = ASSETS_DIR / 'checkpoints' / 'bc_alignment'

    # Task-specific parameters
    task_name = 'pick_up_shoe'

    # Policy parameters
    checkpoint_name = 'best.pt'  # Update with your checkpoint name
    k_visualize = 15  # Number of waypoints to visualize
    k_track = 3  # Number of waypoints to track with controller (open-loop horizon)

    # Simulated starting pose (in practice, read from robot)
    # This is an example pose - replace with actual robot starting pose
    T_WE_start = np.array([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.6],
        [0.0, 0.0, 0.0, 1.0]
    ])

    print("="*80)
    print("BC-Ret Deployment (BC Alignment + Retrieval Interaction)")
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

    save_path = vis_dir / 'bc_ret_test_scene.png'
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
    # Step 3: Retrieve similar demonstration
    # -------------------------------------------------------------------------
    print("\nStep 3: Retrieve demonstration via hierarchical retrieval")
    retrieval = HierarchicalRetrieval(
        T_WC_demo=T_WC,
        T_WC_live=T_WC,
        learned_tasks_dir=str(demo_dir)
    )

    # Retrieve most similar demo based on visual similarity
    retrieved_demo_name = retrieval.get_most_similar_demo_name(
        scene_state=live_scene_state,
        template_task_description=task_name
    )

    print(f"  Retrieved: {retrieved_demo_name}")

    # -------------------------------------------------------------------------
    # Step 4: Load retrieved demonstration
    # -------------------------------------------------------------------------
    print("\nStep 4: Load retrieved demonstration")
    demo_path = demo_dir / retrieved_demo_name

    # Load workspace images
    demo_rgb = np.array(Image.open(str(demo_path / 'head_camera_ws_rgb.png')))
    demo_depth = np.array(Image.open(str(demo_path / 'head_camera_ws_depth_to_rgb.png')))
    demo_segmap = np.load(str(demo_path / 'head_camera_ws_segmap.npy'))
    demo_intrinsics = np.load(str(demo_path / 'head_camera_rgb_intrinsic_matrix.npy'))

    demo_scene_state = SceneState.initialise_from_dict({
        'rgb': demo_rgb,
        'depth': demo_depth,
        'segmap': demo_segmap,
        'intrinsic_matrix': demo_intrinsics
    })

    # Process demonstration segmentation
    demo_scene_state.erode_segmap()
    demo_scene_state.crop_object_using_segmap()

    # Visualize retrieval results
    retrieval_vis_path = vis_dir / 'bc_ret_retrieval.png'
    visualize_retrieval(test_rgb, live_scene_state.segmap, demo_rgb, demo_scene_state.segmap, retrieval_vis_path)

    # Visualize point clouds comparison
    visualize_point_clouds(live_scene_state.o3d_pcd, demo_scene_state.o3d_pcd)

    # Load demonstration bottleneck pose (target for alignment)
    demo_bottleneck_pose = np.load(str(demo_path / 'bottleneck_pose.npy'))
    print(f"  Loaded bottleneck pose from demo")

    # =========================================================================
    # PART 1: ALIGNMENT - BC policy predicts trajectory to bottleneck
    # =========================================================================

    print("\n\nPART 1: ALIGNMENT - BC POLICY")
    print("="*80)

    # -------------------------------------------------------------------------
    # Step 5: Load alignment policy
    # -------------------------------------------------------------------------
    print("\nStep 5: Load trained alignment policy")

    checkpoint_path = checkpoint_dir / checkpoint_name
    if not checkpoint_path.exists():
        print(f"\n  WARNING: Checkpoint not found at {checkpoint_path}")
        print("  This is expected if you haven't trained the alignment policy yet.")
        print("  Skipping policy inference...")
        print("\n  To train the alignment policy:")
        print("    1. Preprocess demonstrations: make preprocess_demos")
        print("    2. Create training dataset: python thousand_tasks/training/act_bn_reaching/create_dataset.py")
        print("    3. Train policy: python thousand_tasks/training/act_bn_reaching/train.py")
        print("    4. Copy checkpoint to assets/checkpoints/bc_alignment/")
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
    # Step 6: Initialize CLIP for language conditioning
    # -------------------------------------------------------------------------
    print("\nStep 6: Initialize language features")

    # Initialize CLIP for text encoding
    clip_model, _ = clip.load("ViT-B/32", device='cpu')

    # Encode task name
    text_tokens = clip.tokenize([task_name.replace('_', ' ')])
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()

    print(f"  Encoded task: '{task_name.replace('_', ' ')}'")

    # -------------------------------------------------------------------------
    # Step 7: Prepare input for policy
    # -------------------------------------------------------------------------
    print("\nStep 7: Prepare input for BC alignment policy")

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
    # Format: [vx, vy, vz, wx, wy, wz, terminate]
    past_actions = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * mt_act_args_dict['history_size']
    )

    # Simulated current robot state
    gripper_state = 0  # Closed

    # -------------------------------------------------------------------------
    # Step 8: Run policy inference
    # -------------------------------------------------------------------------
    print("\nStep 8: Run policy inference to predict alignment trajectory")

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

    print(f"  Policy predicted {len(waypoints)} waypoints to reach bottleneck pose")
    print(f"  First waypoint position (relative to start): {(waypoints[0] @ np.linalg.inv(T_WE_start))[:3, 3]}")

    # -------------------------------------------------------------------------
    # Step 9: Visualize predicted trajectory
    # -------------------------------------------------------------------------
    print("\nStep 9: Visualize predicted alignment trajectory")

    visualize_predicted_trajectory(
        pcd_world=pcd_world,
        pcd_rgb=pcd_rgb,
        waypoints=waypoints,
        T_WE_start=T_WE_start,
        bottleneck_pose=demo_bottleneck_pose,
        k_visualize=k_visualize,
        save_path=vis_dir / 'bc_ret_alignment_trajectory.png'
    )

    print("\n" + "="*80)
    print("ALIGNMENT PHASE INSTRUCTIONS")
    print("="*80)
    print(f"Track the first {k_track} waypoints with a controller:")
    print(f"  1. Use a Cartesian controller to track waypoints in world frame")
    print(f"  2. After tracking {k_track} waypoints, run inference again with:")
    print(f"     - Updated point cloud observation")
    print(f"     - Updated end-effector pose")
    print(f"     - Updated action history")
    print(f"  3. Continue until terminate_prob > 0.95 (reached bottleneck)")
    print(f"  4. Once at bottleneck, proceed to interaction phase")
    print("="*80)

    # Print first few waypoint details
    print(f"\nFirst {k_track} waypoints to track:")
    for i in range(min(k_track, len(waypoints))):
        print(f"  Waypoint {i+1}:")
        print(f"    Position: {waypoints[i][:3, 3]}")
        print(f"    Terminate prob: {terminate_prob[i]:.3f}")

    # =========================================================================
    # PART 2: INTERACTION - Replay demonstrated velocities
    # =========================================================================

    print("\n\nPART 2: INTERACTION - RETRIEVAL-BASED")
    print("="*80)

    # -------------------------------------------------------------------------
    # Step 10: Load demonstration end-effector twists
    # -------------------------------------------------------------------------
    print("\nStep 10: Load demonstrated end-effector velocities")

    demo_eef_twists = np.load(str(demo_path / 'demo_eef_twists.npy'))
    print(f"  Loaded {demo_eef_twists.shape[0]} velocity commands (shape: {demo_eef_twists.shape})")
    print(f"  Format: [vx, vy, vz, wx, wy, wz, gripper_next] per timestep")
    print(f"  Gripper convention: 1 = close, 0 = open")

    # -------------------------------------------------------------------------
    # Step 11: Simulate reaching bottleneck pose
    # -------------------------------------------------------------------------
    print("\nStep 11: [SIMULATED] Assume robot has reached bottleneck pose")
    print("  In practice: Continue tracking waypoints until terminate_prob > 0.95")
    print(f"  Final position would be at: {waypoints[-1][:3, 3]}")

    # -------------------------------------------------------------------------
    # Final instructions
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("INTERACTION PHASE INSTRUCTIONS")
    print("="*80)
    print("Replay demonstrated end-effector velocities:")
    print("  1. Starting from bottleneck pose, apply velocities in end-effector frame")
    print("  2. Update gripper state according to gripper_next column:")
    print("     - 1 = close gripper")
    print("     - 0 = open gripper")
    print("  3. Apply velocities at the demonstrated frequency (typically 30Hz)")
    print("  4. Continue until all velocity commands have been executed")
    print("="*80)

    print(f"\nFirst 5 velocity commands:")
    for i in range(min(5, len(demo_eef_twists))):
        print(f"  Step {i+1}:")
        print(f"    Linear vel (m/s): {demo_eef_twists[i, :3]}")
        print(f"    Angular vel (rad/s): {demo_eef_twists[i, 3:6]}")
        print(f"    Gripper next: {demo_eef_twists[i, 6]:.1f}")


if __name__ == '__main__':
    main()
