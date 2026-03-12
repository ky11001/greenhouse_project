"""
Ret-BC (Retrieval + Behavioral Cloning for Interaction) Deployment Script

Demonstrates the Ret-BC pipeline:
1. Load test image and pre-computed segmentation from assets/inference_example/
2. Retrieve similar demonstration via hierarchical retrieval
3. Estimate relative pose with PointNet++ (shows registration result)
4. Refine pose with Generalized ICP (shows registration result again)
5. Apply 4DOF inductive bias (constrain to tabletop manipulation)
6. Transform demonstration bottleneck pose to live scene
7. [SIMULATE] Assume robot reaches bottleneck pose with motion planning
8. Transform live object point cloud to end-effector frame
9. Run interaction policy (ACT) to predict trajectory
10. Visualize predicted trajectory relative to object and alignment pose

For actual robot deployment:
- Replace test image loading with live camera capture
- Provide pre-computed segmentation masks for target objects
- Use motion planning (e.g., MoveIt, OMPL) to reach bottleneck pose
- Track predicted waypoints with a controller at the recorded frequency
- Update gripper state according to predicted probabilities
"""

import numpy as np
import torch
import clip
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from thousand_tasks.core.globals import ASSETS_DIR
from thousand_tasks.core.utils.scene_state import SceneState
from thousand_tasks.core.utils.se3_tools import pose_inv, rot2euler, euler2rot
from thousand_tasks.retrieval.hierarchical_retrieval import HierarchicalRetrieval
from thousand_tasks.perception.pose_estimation.pnet_4dof_pose_regressor import PointnetPoseRegressor_4dof
from thousand_tasks.perception.pose_estimation.icp_6dof_pose_estimation_refinement import Open3dIcpPoseRefinement
from thousand_tasks.training.act_interaction.policy import ACTPolicy, get_action
from thousand_tasks.training.act_interaction.config import mt_act_args_dict
from thousand_tasks.training.point_cloud_utils import transform_pcd_np, downsample_pcd


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


def apply_4dof_inductive_bias(W_T_delta_6dof: np.ndarray, T_WE: np.ndarray) -> np.ndarray:
    """
    Apply 4DOF inductive bias to transformation (constrain to 4DOF: x, y, z, yaw).

    This function constrains a 6DOF transformation to only 4DOF by:
    1. Removing roll and pitch rotations (keeping only yaw around vertical axis)
    2. Adjusting translation to compensate for the rotation constraint

    The inductive bias is useful for tabletop manipulation tasks where objects
    typically only rotate around the vertical (z) axis, not around x or y axes.
    This reduces the search space and improves robustness.

    Args:
        W_T_delta_6dof: 4x4 transformation matrix (demo → live) in world frame
        T_WE: 4x4 end-effector pose in world frame (used for rotation center)

    Returns:
        W_T_delta_4dof: 4x4 transformation constrained to 4DOF (x, y, z, yaw)

    Mathematical explanation:
    -------------------------
    For tabletop tasks, we want to preserve:
    - Full translation (x, y, z)
    - Yaw rotation (θz around vertical axis)

    But remove:
    - Roll rotation (θx around x-axis)
    - Pitch rotation (θy around y-axis)

    The key insight is that when we remove roll/pitch, we must adjust the
    translation to keep the end-effector at the correct position. This is
    done by:
    1. Computing translation if we rotate around end-effector: R_6dof @ t_E
    2. Computing translation if we rotate around origin with 4DOF: R_4dof @ t_E
    3. Adding the difference to compensate: t_4dof = t_6dof + (R_6dof - R_4dof) @ t_E

    Note: We use a point 24cm above the end-effector (gripper tip) as the
    rotation center to better match the contact point with objects.
    """
    # Create a copy of end-effector pose and move 24cm up (to gripper tip)
    # This accounts for the offset between wrist and actual contact point
    T_WE_copy = T_WE.copy()
    T_WE_copy[:3, 3] += T_WE_copy[:3, :3] @ np.array([0, 0, 0.24])
    t_WE = T_WE_copy[:3, 3:]  # Position of rotation center

    # Extract rotation and translation from 6DOF transformation
    W_R_delta_6dof = W_T_delta_6dof[:3, :3]
    W_t_delta_6dof = W_T_delta_6dof[:3, 3:]

    # Convert 6DOF rotation to Euler angles (xyz convention)
    three_dof_euler = rot2euler('xyz', W_R_delta_6dof, degrees=False)

    # Zero out roll (θx) and pitch (θy), keeping only yaw (θz)
    # This constrains rotation to vertical axis only
    three_dof_euler[:2] = 0

    # Convert back to rotation matrix (now only yaw rotation)
    W_R_delta_4dof = euler2rot('xyz', three_dof_euler, degrees=False)

    # Adjust translation to compensate for rotation constraint
    # Formula: t_new = t_old + (R_old - R_new) @ rotation_center
    # This ensures the end-effector still reaches the same world position
    # even though we've constrained the rotation
    W_t_delta_4dof = W_R_delta_6dof @ t_WE + W_t_delta_6dof - W_R_delta_4dof @ t_WE

    # Construct 4DOF transformation matrix
    W_T_delta_4dof = np.eye(4)
    W_T_delta_4dof[:3, :3] = W_R_delta_4dof
    W_T_delta_4dof[:3, 3:] = W_t_delta_4dof

    return W_T_delta_4dof


def visualize_predicted_trajectory(pcd_world, pcd_rgb, waypoints, T_WE_bottleneck, k_visualize=10, save_path=None):
    """
    Visualize predicted trajectory in Open3D with future steps faded out.

    Args:
        pcd_world: Object point cloud in world frame (N x 3)
        pcd_rgb: RGB colors for point cloud (N x 3, range 0-1)
        waypoints: List of predicted SE(3) poses (4x4 matrices)
        T_WE_bottleneck: Bottleneck pose (4x4 matrix) - starting position
        k_visualize: Number of future waypoints to visualize
        save_path: Optional path to save visualization screenshot
    """
    import open3d as o3d

    # Create point cloud
    o3d_pcd = o3d.geometry.TriangleMesh()
    o3d_pcd_points = o3d.geometry.PointCloud()
    o3d_pcd_points.points = o3d.utility.Vector3dVector(pcd_world)
    o3d_pcd_points.colors = o3d.utility.Vector3dVector(pcd_rgb)

    # Create visualizations list
    geometries = [o3d_pcd_points]

    # Add bottleneck pose (bright, full opacity)
    coord_frame_bottleneck = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.08, origin=[0, 0, 0])
    coord_frame_bottleneck.transform(T_WE_bottleneck)
    geometries.append(coord_frame_bottleneck)

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
        trajectory_points = [T_WE_bottleneck[:3, 3]] + [wp[:3, 3] for wp in waypoints[:k_visualize]]
        lines = [[i, i+1] for i in range(len(trajectory_points)-1)]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(trajectory_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])  # Red trajectory
        geometries.append(line_set)

    # Visualize
    print(f"  Displaying predicted trajectory (showing {num_waypoints} future waypoints)...")
    print("  Close window to continue...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Ret-BC: Predicted Trajectory Relative to Object",
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
    checkpoint_dir = ASSETS_DIR / 'checkpoints' / 'bc_interaction'

    # Task-specific parameters
    task_name = 'pick_up_shoe'

    # Policy parameters
    checkpoint_name = 'best.pt'  # Update with your checkpoint name
    k_visualize = 10  # Number of future waypoints to visualize
    k_track = 3  # Number of waypoints to track with controller (open-loop horizon)

    print("="*80)
    print("Ret-BC Deployment (Retrieval + BC Interaction)")
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

    save_path = vis_dir / 'ret_bc_test_scene.png'
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
    # Step 4: Load and segment demonstration
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
    retrieval_vis_path = vis_dir / 'ret_bc_retrieval.png'
    visualize_retrieval(test_rgb, live_scene_state.segmap, demo_rgb, demo_scene_state.segmap, retrieval_vis_path)

    # Visualize point clouds comparison
    visualize_point_clouds(live_scene_state.o3d_pcd, demo_scene_state.o3d_pcd)

    # =========================================================================
    # PART 1: ALIGNMENT - Estimate target pose for robot
    # =========================================================================

    # -------------------------------------------------------------------------
    # Step 5: Estimate relative pose with PointNet++
    # -------------------------------------------------------------------------
    print("\nStep 5: Estimate relative pose with PointNet++")
    pose_estimator = PointnetPoseRegressor_4dof(
        filter_pointcloud=True,
        n_points=2048,
        T_WC=T_WC,
        T_WC_demo=T_WC,
        depth_units='mm',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # Estimate transformation from demo to live scene (in world frame)
    W_T_delta = pose_estimator.estimate_relative_pose(
        scene1_state=demo_scene_state,
        scene2_state=live_scene_state,
        visualise_pcds=True,
        verbose=False
    )

    print(f"  PointNet++ prediction complete")

    # -------------------------------------------------------------------------
    # Step 5b: Refine pose estimate with Generalized ICP
    # -------------------------------------------------------------------------
    print("\nStep 5b: Refine pose with Generalized ICP")
    print("  Initializing ICP refinement...")

    pose_refiner = Open3dIcpPoseRefinement(
        error_metric='generalised-icp',
        max_correspondence_distance=0.1,
        max_iteration=20,
        depth_units='mm',
        timeout=3
    )

    # Convert world-frame transformation to camera frame for ICP
    C_T_delta = pose_inv(T_WC) @ W_T_delta @ T_WC

    # Refine pose using Generalized ICP
    C_T_delta_refined = pose_refiner.refine_relative_pose(
        scene1_state=demo_scene_state,
        scene2_state=live_scene_state,
        T_delta_init=C_T_delta,
        T_WC_live=T_WC,
        verbose=False,
        visualise_pcds=True
    )

    # Convert refined pose back to world frame
    W_T_delta_refined = T_WC @ C_T_delta_refined @ pose_inv(T_WC)

    print(f"  ICP refinement complete")

    # -------------------------------------------------------------------------
    # Step 6: Apply 4DOF inductive bias
    # -------------------------------------------------------------------------
    print("\nStep 6: Apply 4DOF inductive bias")

    # Load demonstration bottleneck pose to get rotation center
    demo_bottleneck_pose = np.load(str(demo_path / 'bottleneck_pose.npy'))

    # Apply 4DOF constraint (x, y, z, yaw only - no roll/pitch)
    W_T_delta_4dof = apply_4dof_inductive_bias(
        W_T_delta_6dof=W_T_delta_refined,
        T_WE=demo_bottleneck_pose
    )

    print(f"  4DOF constraint applied (removed roll and pitch)")

    # -------------------------------------------------------------------------
    # Step 7: Transform demonstration bottleneck pose to live scene
    # -------------------------------------------------------------------------
    print("\nStep 7: Transform bottleneck pose to live scene")

    # Transform demonstration bottleneck pose to live scene
    live_bottleneck_pose = W_T_delta_4dof @ demo_bottleneck_pose

    print(f"  Demo bottleneck pose (T_WE):\n{demo_bottleneck_pose}")
    print(f"  Live bottleneck pose (T_WE):\n{live_bottleneck_pose}")
    print("\n" + "="*80)
    print("ALIGNMENT PHASE COMPLETE")
    print("="*80)
    print("Target end-effector pose for live scene:")
    print(f"{live_bottleneck_pose}")
    print("\nIn practice: Use motion planning (e.g., MoveIt, OMPL) or a linear")
    print("controller to move the robot's end-effector to this bottleneck pose.")
    print("="*80)

    # =========================================================================
    # PART 2: INTERACTION - Predict trajectory with BC policy
    # =========================================================================

    print("\n\nPART 2: INTERACTION - BC POLICY")
    print("="*80)

    # -------------------------------------------------------------------------
    # Step 8: Load interaction policy
    # -------------------------------------------------------------------------
    print("\nStep 8: Load trained interaction policy")

    checkpoint_path = checkpoint_dir / checkpoint_name
    if not checkpoint_path.exists():
        print(f"\n  WARNING: Checkpoint not found at {checkpoint_path}")
        print("  This is expected if you haven't trained the interaction policy yet.")
        print("  Skipping policy inference...")
        print("\n  To train the interaction policy:")
        print("    1. Preprocess demonstrations: make preprocess_demos")
        print("    2. Create training dataset: python thousand_tasks/training/act_interaction/create_dataset.py")
        print("    3. Train policy: python thousand_tasks/training/act_interaction/train.py")
        print("    4. Copy checkpoint to assets/checkpoints/")
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
    # Step 9: Initialize CLIP for language conditioning
    # -------------------------------------------------------------------------
    print("\nStep 9: Initialize language features")

    # Initialize CLIP for text encoding
    clip_model, _ = clip.load("ViT-B/32", device='cpu')

    # Encode task name
    text_tokens = clip.tokenize([task_name.replace('_', ' ')])
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()

    print(f"  Encoded task: '{task_name.replace('_', ' ')}'")

    # -------------------------------------------------------------------------
    # Step 10: Simulate reaching bottleneck pose
    # -------------------------------------------------------------------------
    print("\nStep 10: [SIMULATED] Assume robot reaches bottleneck pose")
    print("  In practice: Motion planner would move robot to live_bottleneck_pose")
    print("  For this example: We assume the robot is now at the target pose")

    # Simulate that we're now at the bottleneck pose
    T_WE_current = live_bottleneck_pose
    gripper_state = 0  # Closed

    # -------------------------------------------------------------------------
    # Step 11: Transform object point cloud to end-effector frame
    # -------------------------------------------------------------------------
    print("\nStep 11: Transform object point cloud to end-effector frame")

    # Get object point cloud in world frame
    o3d_pcd_world = live_scene_state.o3d_pcd
    pcd_world = np.asarray(o3d_pcd_world.points)
    pcd_rgb = np.asarray(o3d_pcd_world.colors)

    # Downsample to match training
    pcd_world, pcd_rgb = downsample_pcd(pcd_world, pcd_rgb, 2048)

    # Transform to end-effector frame (note: this is done inside get_action)
    # But we keep world frame for visualization

    # Update Open3D point cloud
    o3d_pcd_world.points = o3d.utility.Vector3dVector(pcd_world)
    o3d_pcd_world.colors = o3d.utility.Vector3dVector(pcd_rgb)

    print(f"  Point cloud has {len(pcd_world)} points after downsampling")

    # -------------------------------------------------------------------------
    # Step 12: Run policy inference
    # -------------------------------------------------------------------------
    print("\nStep 12: Run policy inference to predict trajectory")

    # Initialize action history (for temporal context)
    past_actions = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]] * mt_act_args_dict['history_size']
    )  # [vx, vy, vz, wx, wy, wz, gripper, terminate]

    # Run inference
    with torch.no_grad():
        waypoints, gripper_action_prob, terminate_prob, _ = get_action(
            policy=policy,
            o3d_pcd_world=o3d_pcd_world,
            T_WE=T_WE_current,
            gripper_state=gripper_state,
            text_features=text_features,
            past_actions=past_actions,
            plotter=None,
            shown=False
        )

    print(f"  Policy predicted {len(waypoints)} waypoints")
    print(f"  First waypoint position (relative to current): {(waypoints[0] @ np.linalg.inv(T_WE_current))[:3, 3]}")

    # -------------------------------------------------------------------------
    # Step 13: Visualize predicted trajectory
    # -------------------------------------------------------------------------
    print("\nStep 13: Visualize predicted trajectory")

    visualize_predicted_trajectory(
        pcd_world=pcd_world,
        pcd_rgb=pcd_rgb,
        waypoints=waypoints,
        T_WE_bottleneck=T_WE_current,
        k_visualize=k_visualize,
        save_path=vis_dir / 'ret_bc_trajectory.png'
    )

    # -------------------------------------------------------------------------
    # Final instructions
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("INTERACTION PHASE INSTRUCTIONS")
    print("="*80)
    print(f"Track the first {k_track} waypoints with a controller:")
    print(f"  1. Use a Cartesian controller to track waypoints in world frame")
    print(f"  2. Update gripper state based on gripper_action_prob (gripper closure probability):")
    print(f"     - Open gripper if prob < 0.1 (low closure = should be open)")
    print(f"     - Close gripper if prob > 0.9 (high closure = should be closed)")
    print(f"  3. After tracking {k_track} waypoints, run inference again with:")
    print(f"     - Updated point cloud observation")
    print(f"     - Updated end-effector pose")
    print(f"     - Updated action history")
    print(f"  4. Continue until terminate_prob > 0.85")
    print("="*80)

    # Print first few waypoint details
    print(f"\nFirst {k_track} waypoints to track:")
    for i in range(min(k_track, len(waypoints))):
        print(f"  Waypoint {i+1}:")
        print(f"    Position: {waypoints[i][:3, 3]}")
        print(f"    Gripper prob: {gripper_action_prob[i]:.3f}")
        print(f"    Terminate prob: {terminate_prob[i]:.3f}")


if __name__ == '__main__':
    main()
