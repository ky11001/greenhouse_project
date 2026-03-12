"""
Demo Preprocessing Script for BC Training

This script preprocesses raw demonstration data for BC policy training by:
1. Loading pre-computed workspace segmentation (first frame) to seed XMem
2. Tracking objects across all timesteps using XMem
3. Saving per-timestep segmentation masks

IMPORTANT DISTINCTION:
======================
- MT3 (deployment): Only needs workspace segmentation (first frame) for retrieval
- BC Training: Needs segmentation masks for EVERY timestep to train the policy

The provided demonstrations in assets/demonstrations/ MUST include:
  - head_camera_ws_segmap.npy: Pre-computed workspace segmentation (first frame) - FOR MT3 & XMem seed
  - head_camera_rgb.npy: Full RGB-D video (T timesteps)
  - head_camera_depth.npy: Full depth video (T timesteps)
  - bottleneck_pose.npy: Target end-effector pose
  - demo_eef_twists.npy: Demonstrated velocities

NOTE: For your own demonstrations, you must segment the first frame manually or with
      segmentation tools (LangSAM, SAM, etc.) and save as head_camera_ws_segmap.npy
      before running this script.

This script generates:
  - head_camera_masks.npy: Per-timestep segmentation masks (T, H, W) - FOR BC TRAINING
  - demo_video.mp4: Video visualization of RGB trajectory
  - demo_segmented_video.mp4: Video visualization with segmentation overlay

Usage:
======
python thousand_tasks/demo_preprocessing/preprocess_demos.py

Or via Make:
============
make preprocess_demos
"""

import os
import time
from os import mkdir
from os.path import join

import numpy as np
from tqdm import tqdm
import imageio

from thousand_tasks.core.xmem.xmem_wrapper import XMemTracker
from thousand_tasks.core.globals import ASSETS_DIR

# ============================================================================
# Configuration: Update task names for your demonstrations
# ============================================================================

# Directory containing demonstrations
TASKS_DIR = ASSETS_DIR / 'demonstrations'

# Task names to process (each must have head_camera_ws_segmap.npy already present)
TASK_NAMES = [f'demo_task_full_{i:03d}' for i in range(32, 42)]

# Camera configuration
CAMERA_NAME = 'head_camera'
T_WC_PATH = ASSETS_DIR / 'T_WC_head.npy'

# Processing options
SAVE_VIDEOS = True  # Save MP4 videos of demonstrations
SAVE_PNG_IMAGES = False  # Save individual PNG frames (for debugging)
VIDEO_FPS = 30  # Frames per second for video output
DEVICE = 'cuda'  # 'cuda' or 'cpu'

# ============================================================================


def create_video_from_frames(frames, output_path, fps=10):
    import cv2
    if not frames:
        return
    height, width, _ = frames[0].shape
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # OpenCV uses BGR, so convert from RGB
        out.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    out.release()


def create_segmented_video(rgb_frames, masks, output_path, fps=10, alpha=0.5):
    """
    Create MP4 video with segmentation overlay.

    Args:
        rgb_frames: List of numpy arrays (H, W, 3) with values in [0, 255]
        masks: List of boolean numpy arrays (H, W)
        output_path: Path to save the MP4 file
        fps: Frames per second
        alpha: Transparency of overlay (0=transparent, 1=opaque)
    """
    segmented_frames = []

    for rgb, mask in zip(rgb_frames, masks):
        # Create overlay: green highlight on segmented regions
        overlay = rgb.copy().astype(np.float32)
        # Expand mask to 3 channels: (H, W) -> (H, W, 3) for broadcasting
        mask_3ch = mask[..., None]
        # Apply green overlay where mask is True
        overlay = np.where(mask_3ch, overlay * (1 - alpha) + np.array([0, 255, 0]) * alpha, overlay)
        segmented_frames.append(overlay.astype(np.uint8))

    # Write video
    writer = imageio.get_writer(output_path, fps=fps, format='FFMPEG', mode='I', codec='libx264')
    for frame in segmented_frames:
        writer.append_data(frame)
    writer.close()


def preprocess_demonstrations():
    """
    Preprocess all configured demonstrations by generating per-timestep segmentation masks.

    REQUIREMENTS:
    =============
    Each demonstration must already have a pre-computed workspace segmentation mask:
        - {CAMERA_NAME}_ws_segmap.npy: Binary mask (H, W) of the object in the first frame

    For new demonstrations, you must segment the first frame using:
        - LangSAM, SAM, or other segmentation tools
        - Manual annotation
        - Your own segmentation pipeline

    Then save the mask as head_camera_ws_segmap.npy before running this script.

    WHAT THIS SCRIPT DOES:
    ======================
    1. Loads the pre-computed workspace segmentation mask
    2. Uses it to seed XMem video object tracker
    3. Propagates segmentation across all timesteps
    4. Saves per-timestep masks for BC training
    """
    from thousand_tasks.core.utils.demo import Demo
    from thousand_tasks.core.utils.visualisation import plot_image, plot_images_in_grid
    from thousand_tasks.core.utils.segmentation_utils import get_valid_pixels_based_on_workspace

    T_WC = np.load(str(T_WC_PATH))

    print("="*80)
    print("Demo Preprocessing for BC Training")
    print("="*80)
    print(f"Tasks directory: {TASKS_DIR}")
    print(f"Tasks to process: {TASK_NAMES}")
    print(f"Device: {DEVICE}")
    print("="*80)

    for i, task_name in enumerate(TASK_NAMES):
        try:
            print(f'\n[{i + 1}/{len(TASK_NAMES)}] Processing: {task_name}')

            # Initialize XMem tracker
            xmem = XMemTracker(single_object=False, device=DEVICE)

            start = time.time()

            # Create output directory for visualizations
            if SAVE_PNG_IMAGES:
                seg_img_dir = join(TASKS_DIR, task_name, 'segmented_images')
                try:
                    mkdir(seg_img_dir)
                except FileExistsError:
                    pass

            # Load demonstration
            demo = Demo(task_name, str(TASKS_DIR))
            demo.load_rgbd_images(CAMERA_NAME)
            demo.load_workspace_images(CAMERA_NAME)
            intrinsic_matrix = demo.intrinsic_matrices[CAMERA_NAME]

            # Step 1: Load pre-computed workspace segmentation mask
            print(f'  [1/3] Loading pre-computed workspace segmentation...')
            ws_segmap_path = join(TASKS_DIR, task_name, f'{CAMERA_NAME}_ws_segmap.npy')

            if not os.path.exists(ws_segmap_path):
                raise FileNotFoundError(
                    f"Workspace segmentation not found: {ws_segmap_path}\n"
                    f"Please segment the first frame and save as {CAMERA_NAME}_ws_segmap.npy"
                )

            mask = np.load(ws_segmap_path).astype(np.uint8)
            mask[mask > 0] = 1
            print(f'  Loaded: {CAMERA_NAME}_ws_segmap.npy (shape: {mask.shape})')

            # Refine segmentation mask (remove invalid workspace pixels)
            # print(f'  Refining mask using workspace boundaries...')
            # valid = get_valid_pixels_based_on_workspace(
            #    demo.workspace_depth_images[CAMERA_NAME],
            #    intrinsic_matrix,
            #    T_WC
            # )
            # mask[np.logical_not(valid)] = False

            # Step 2: Initialize XMem with the workspace segmentation
            print(f'  [2/3] Initializing XMem tracker...')
            xmem.initialise(
                rgb=demo.workspace_rgb_images[CAMERA_NAME],
                segmap=mask
            )

            if SAVE_PNG_IMAGES:
                plot_image(
                    demo.workspace_rgb_images[CAMERA_NAME] * mask[..., None],
                    save_path=join(TASKS_DIR, task_name, 'segmented_images', f'a_xmem_seed.png')
                )

            # Step 3: Track object through all timesteps using XMem
            num_frames = len(demo.rgb_images[CAMERA_NAME])
            print(f'  [3/3] Tracking object through {num_frames} timesteps...')
            segmentation_masks = []
            rgb_frames_for_video = []  # Store frames for video generation

            for j in tqdm(range(num_frames), desc='        Tracking', leave=False):
                rgb = demo.rgb_images[CAMERA_NAME][j]
                depth = demo.depth_images[CAMERA_NAME][j]

                # Mask out invalid pixels
                valid = get_valid_pixels_based_on_workspace(depth, intrinsic_matrix, T_WC)
                rgb_temp = rgb.copy()
                rgb_temp[np.logical_not(valid)] = 0

                # Track with XMem
                mask = xmem.compute_object_segmap(rgb)
                # mask[np.logical_not(valid)] = False   
                segmentation_masks.append(mask)

                # Store original RGB frame for video (scale to uint8 if needed)
                if SAVE_VIDEOS:
                    if rgb.dtype == np.float32 or rgb.dtype == np.float64:
                        rgb_uint8 = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
                    else:
                        rgb_uint8 = rgb
                    rgb_frames_for_video.append(rgb_uint8)

                # Save visualization every 5 frames
                if SAVE_PNG_IMAGES and j % 5 == 0:
                    plot_images_in_grid(
                        [rgb_temp, rgb * mask[..., None]],
                        figsize=(7, 2),
                        tight_layout=True,
                        save_path=join(TASKS_DIR, task_name, 'segmented_images', f'segmented_rgb_{j:04d}.png')
                    )

            # Save per-timestep masks (for BC training)
            masks_array = np.array(segmentation_masks)
            np.save(join(TASKS_DIR, task_name, f'{CAMERA_NAME}_masks.npy'), masks_array)
            print(f'  Saved: {CAMERA_NAME}_masks.npy (shape: {masks_array.shape}) - for BC training')

            # Step 4: Generate videos
            if SAVE_VIDEOS:
                print(f'  [4/4] Generating videos...')

                # Create RGB demonstration video
                video_path = join(TASKS_DIR, task_name, 'demo_video.mp4')
                create_video_from_frames(rgb_frames_for_video, video_path, fps=VIDEO_FPS)
                print(f'  Saved: demo_video.mp4')

                # Create segmented demonstration video
                segmented_video_path = join(TASKS_DIR, task_name, 'demo_segmented_video.mp4')
                create_segmented_video(rgb_frames_for_video, segmentation_masks,
                                        segmented_video_path, fps=VIDEO_FPS, alpha=0.4)
                print(f'  Saved: demo_segmented_video.mp4')

            elapsed = time.time() - start
            print(f'  ✓ Completed in {elapsed:.1f}s')

        except Exception as e:
            print(f'  ✗ ERROR: {e}')
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("Preprocessing complete!")
    print("="*80)
    print(f"Processed {len(TASK_NAMES)} demonstrations")
    print(f"Output files per demonstration:")
    print(f"  - {CAMERA_NAME}_masks.npy (for BC training)")
    if SAVE_VIDEOS:
        print(f"  - demo_video.mp4 (RGB trajectory visualization)")
        print(f"  - demo_segmented_video.mp4 (segmentation overlay)")
    if SAVE_PNG_IMAGES:
        print(f"  - segmented_images/ (per-frame PNG visualizations)")
    print("="*80)


if __name__ == '__main__':
    preprocess_demonstrations()
