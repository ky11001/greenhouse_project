import sys
import rospy
import numpy as np
import torch
import intera_interface
from PIL import Image
from pathlib import Path
from tf.transformations import quaternion_from_matrix
from intera_interface import Limb
from scipy.spatial.transform import Rotation

# RealSense
import pyrealsense2 as rs

# SAM
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# MT3
from thousand_tasks.core.globals import ASSETS_DIR
from thousand_tasks.core.utils.scene_state import SceneState
from thousand_tasks.core.utils.se3_tools import pose_inv, rot2euler, euler2rot
from thousand_tasks.perception.pose_estimation.pnet_4dof_pose_regressor import PointnetPoseRegressor_4dof
from thousand_tasks.perception.pose_estimation.icp_6dof_pose_estimation_refinement import Open3dIcpPoseRefinement
from intera_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from geometry_msgs.msg import PoseStamped


# ──────────────────────────────────────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────────────────────────────────────

def capture_live_frame(width=640, height=480, fps=30, warmup=20):
    """Capture a single RGB+depth frame from RealSense."""
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16,  fps)
    pipeline.start(config)
    for _ in range(warmup):
        pipeline.wait_for_frames()
    frames      = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    rgb   = np.asanyarray(color_frame.get_data())[:, :, ::-1].copy()
    depth = np.asanyarray(depth_frame.get_data())
    pipeline.stop()
    return rgb, depth


def segment_auto(rgb, depth, sam_ckpt, min_area=500, max_area_frac=0.3,
                 min_depth_mm=200, max_depth_mm=1500):
    """
    Automatically segment the closest central object using SAM + depth filtering.
    Rejects masks where average depth is outside min/max range.
    """
    h, w = rgb.shape[:2]
    cx, cy = w // 2, h // 2

    print("  Running SAM automatic mask generation...")
    sam = sam_model_registry["vit_h"](checkpoint=str(sam_ckpt))
    mask_gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=min_area
    )
    masks = mask_gen.generate(rgb)
    print(f"  Found {len(masks)} candidate masks")

    best_mask  = None
    best_score = float('inf')
    max_area   = h * w * max_area_frac

    for m in masks:
        area = m['area']
        if area < min_area or area > max_area:
            continue

        # Depth filter — reject masks that are too far or too close
        mask_depth = depth[m['segmentation']]
        valid_depth = mask_depth[mask_depth > 0]
        if len(valid_depth) < 50:
            continue
        avg_depth = valid_depth.mean()
        if avg_depth < min_depth_mm or avg_depth > max_depth_mm:
            print(f"  Skipping mask (avg depth {avg_depth:.0f}mm out of range)")
            continue

        # Pick most central mask within depth range
        ys, xs = np.where(m['segmentation'])
        mx, my = xs.mean(), ys.mean()
        dist = np.sqrt((mx - cx)**2 + (my - cy)**2)
        if dist < best_score:
            best_score = dist
            best_mask  = m['segmentation']

    if best_mask is None:
        raise RuntimeError(
            "No mask found within depth range. "
            "Check min_depth_mm/max_depth_mm match your scene.")

    print(f"  Selected mask: {best_mask.sum()} px, "
          f"centre dist: {best_score:.1f} px")
    return best_mask.astype(bool)


# ──────────────────────────────────────────────────────────────────────────────
# MT3 helpers
# ──────────────────────────────────────────────────────────────────────────────

def apply_4dof_inductive_bias(W_T_delta_6dof, T_WE):
    T_WE_copy = T_WE.copy()
    T_WE_copy[:3, 3] += T_WE_copy[:3, :3] @ np.array([0, 0, 0.24])
    t_WE            = T_WE_copy[:3, 3:]
    W_R_delta_6dof  = W_T_delta_6dof[:3, :3]
    W_t_delta_6dof  = W_T_delta_6dof[:3, 3:]
    three_dof_euler = rot2euler('xyz', W_R_delta_6dof, degrees=False)
    three_dof_euler[:2] = 0
    W_R_delta_4dof  = euler2rot('xyz', three_dof_euler, degrees=False)
    W_t_delta_4dof  = W_R_delta_6dof @ t_WE + W_t_delta_6dof - W_R_delta_4dof @ t_WE
    W_T_delta_4dof  = np.eye(4)
    W_T_delta_4dof[:3, :3] = W_R_delta_4dof
    W_T_delta_4dof[:3, 3:] = W_t_delta_4dof
    return W_T_delta_4dof


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    rospy.init_node('mt3_sawyer_deploy', anonymous=True)

    limb    = Limb('right')
    print("\n[Step 0] Moving to home position...")
    home_joints = {
        'right_j0': 0.0,
        'right_j1': -1.57,
        'right_j2': 0.0,
        'right_j3':  1.57,
        'right_j4': -1.57,
        'right_j5':  0.0,
        'right_j6':  0.0
    }
    limb.set_joint_position_speed(0.1)
    limb.move_to_joint_positions(home_joints, timeout=15.0)
    rospy.sleep(1.0)
    print("  At home position.")
    print("  At observation pose. Capturing scene...")

    demo_path = ASSETS_DIR / 'demonstrations' / 'pick_up_tomato' / 'pick_up_tomato_0000'
    sam_ckpt  = ASSETS_DIR / 'sam_vit_h_4b8939.pth'
    T_WC = np.load(str(ASSETS_DIR / 'T_WC_wrist_home.npy'))
    ik_service = rospy.ServiceProxy(
        'ExternalTools/right/PositionKinematicsNode/IKService', SolvePositionIK)

    vis_dir = ASSETS_DIR / 'example_visualisations'
    vis_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: Capture live scene
    # -------------------------------------------------------------------------
    print("\n[Step 1] Capturing live scene from RealSense...")
    live_rgb, live_depth = capture_live_frame()
    print(f"  RGB: {live_rgb.shape}  "
          f"Depth: {live_depth[live_depth>0].min()}–{live_depth[live_depth>0].max()}mm")
    Image.fromarray(live_rgb).save(str(vis_dir / 'live_scene.png'))
    print(f"  Saved to: {vis_dir / 'live_scene.png'}")

    # -------------------------------------------------------------------------
    # STEP 2: Auto-segment live scene
    # -------------------------------------------------------------------------
    print("\n[Step 2] Auto-segmenting with SAM...")
    live_segmap = segment_auto(live_rgb, live_depth, sam_ckpt,
                           min_depth_mm=400, max_depth_mm=900)
    intrinsics  = np.load(str(demo_path / 'head_camera_rgb_intrinsic_matrix.npy'))

    # Save segmentation preview
    seg_preview = live_rgb.copy()
    seg_preview[~live_segmap] = (seg_preview[~live_segmap] * 0.3).astype(np.uint8)
    Image.fromarray(seg_preview).save(str(vis_dir / 'live_segmentation.png'))
    print(f"  Saved segmentation preview to: {vis_dir / 'live_segmentation.png'}")

    live_scene_state = SceneState.initialise_from_dict({
        'rgb': live_rgb, 'depth': live_depth,
        'segmap': live_segmap, 'intrinsic_matrix': intrinsics
    })
    live_scene_state.erode_segmap()
    live_scene_state.crop_object_using_segmap()
    # DEBUG
    print("  Live segmap True pixels:", live_segmap.sum())
    print("  Live depth in segmap - valid pixels:", (live_depth[live_segmap] > 0).sum())
    print("  Live depth in segmap - range:", 
        live_depth[live_segmap & (live_depth > 0)].min() if (live_depth[live_segmap] > 0).any() else "NO VALID DEPTH",
        "-",
        live_depth[live_segmap].max())
    print("  Live point cloud size:", len(live_scene_state.o3d_pcd.points))

    # -------------------------------------------------------------------------
    # STEP 3: Load demo
    # -------------------------------------------------------------------------
    print(f"\n[Step 3] Loading demo...")
    demo_rgb        = np.array(Image.open(str(demo_path / 'head_camera_ws_rgb.png')))
    demo_depth      = np.array(Image.open(str(demo_path / 'head_camera_ws_depth_to_rgb.png')))
    demo_segmap     = np.load(str(demo_path / 'head_camera_ws_segmap.npy'))
    demo_intrinsics = np.load(str(demo_path / 'head_camera_rgb_intrinsic_matrix.npy'))

    demo_scene_state = SceneState.initialise_from_dict({
        'rgb': demo_rgb, 'depth': demo_depth,
        'segmap': demo_segmap, 'intrinsic_matrix': demo_intrinsics
    })
    demo_scene_state.erode_segmap()
    demo_scene_state.crop_object_using_segmap()
    print("  Demo point cloud size:", len(demo_scene_state.o3d_pcd.points))

    # -------------------------------------------------------------------------
    # STEP 4: Pose estimation
    # -------------------------------------------------------------------------
    print("\n[Step 4] Estimating relative pose...")
    pose_estimator = PointnetPoseRegressor_4dof(
        filter_pointcloud=True, n_points=2048,
        T_WC=T_WC, T_WC_demo=T_WC,
        depth_units='mm',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    W_T_delta = pose_estimator.estimate_relative_pose(demo_scene_state, live_scene_state)

    pose_refiner = Open3dIcpPoseRefinement(
        error_metric='generalised-icp',
        max_correspondence_distance=0.05,
        max_iteration=30,
        timeout=1.0
    )
    C_T_delta         = pose_inv(T_WC) @ W_T_delta @ T_WC
    C_T_delta_refined = pose_refiner.refine_relative_pose(
                            demo_scene_state, live_scene_state,
                            T_delta_init=C_T_delta, T_WC_live=T_WC)
    W_T_delta_refined = T_WC @ C_T_delta_refined @ pose_inv(T_WC)

    demo_bottleneck_pose = np.load(str(demo_path / 'bottleneck_pose.npy'))
    W_T_delta_4dof       = apply_4dof_inductive_bias(W_T_delta_refined, demo_bottleneck_pose)
    live_bottleneck_pose = W_T_delta_4dof @ demo_bottleneck_pose

    print(f"  Live bottleneck position (x,y,z): {live_bottleneck_pose[:3,3].round(4)}")

    # -------------------------------------------------------------------------
    # STEP 5: Move to bottleneck pose
    # -------------------------------------------------------------------------
    print("\n[Step 5] Moving to bottleneck pose...")

    ik_request   = SolvePositionIKRequest()
    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = 'base'
    pose_stamped.header.stamp    = rospy.Time.now()
    pose_stamped.pose.position.x = live_bottleneck_pose[0, 3]
    pose_stamped.pose.position.y = live_bottleneck_pose[1, 3]
    pose_stamped.pose.position.z = live_bottleneck_pose[2, 3]
    q = quaternion_from_matrix(live_bottleneck_pose)
    pose_stamped.pose.orientation.x = q[0]
    pose_stamped.pose.orientation.y = q[1]
    pose_stamped.pose.orientation.z = q[2]
    pose_stamped.pose.orientation.w = q[3]
    ik_request.tip_names.append('right_hand')
    ik_request.pose_stamp.append(pose_stamped)

    ik_response = ik_service(ik_request)
    if ik_response.result_type[0] > 0:
        jnames  = ik_response.joints[0].name
        jangles = ik_response.joints[0].position
        limb.set_joint_position_speed(0.1)
        limb.move_to_joint_positions(dict(zip(jnames, jangles)), timeout=15.0)
        print("  Reached bottleneck pose.")
    else:
        print("  !! IK failed. Exiting.")
        return

    # -------------------------------------------------------------------------
    # STEP 6: Replay interaction twists
    # -------------------------------------------------------------------------
    print("\n[Step 6] Replaying interaction twists...")
    twists = np.load(str(demo_path / 'demo_eef_twists.npy'))
    print(f"  {twists.shape} at 30Hz = {len(twists)/30:.1f}s")

    dt = 1.0 / 30.0

    cur_pose = limb.endpoint_pose()
    cur_pos  = np.array([cur_pose['position'].x,
                         cur_pose['position'].y,
                         cur_pose['position'].z])
    cur_quat = cur_pose['orientation']
    R        = Rotation.from_quat([cur_quat.x, cur_quat.y,
                                   cur_quat.z, cur_quat.w])

    for i, twist in enumerate(twists):
        if rospy.is_shutdown():
            break

        v_base  = R.as_matrix() @ twist[:3]
        cur_pos = cur_pos + v_base * dt

        ps = PoseStamped()
        ps.header.frame_id = 'base'
        ps.header.stamp    = rospy.Time.now()
        ps.pose.position.x = cur_pos[0]
        ps.pose.position.y = cur_pos[1]
        ps.pose.position.z = cur_pos[2]
        ps.pose.orientation.x = q[0]
        ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]
        ps.pose.orientation.w = q[3]

        ik_req = SolvePositionIKRequest()
        ik_req.tip_names.append('right_hand')
        ik_req.pose_stamp.append(ps)
        ik_resp = ik_service(ik_req)

        if ik_resp.result_type[0] > 0:
            jnames  = ik_resp.joints[0].name
            jangles = ik_resp.joints[0].position
            limb.set_joint_positions(dict(zip(jnames, jangles)))

        if len(twist) > 6:
            if twist[6] > 0.5:
                gripper.close()
            else:
                gripper.open()

        if i % 10 == 0:
            print(f"  ... {i}/{len(twists)}  pos={cur_pos.round(3)}")

        rospy.sleep(dt)

    limb.set_joint_velocities(dict(zip(limb.joint_names(), [0.0] * 7)))
    print("\n  Interaction complete.")
    print("Task complete.")


if __name__ == '__main__':
    main()