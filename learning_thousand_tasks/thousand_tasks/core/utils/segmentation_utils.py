"""
Segmentation utility functions for workspace validation and mask refinement.
"""

import numpy as np
from PIL import ImageFilter, Image

from thousand_tasks.core.utils.img_to_pcd import img_to_pcd
from thousand_tasks.core.utils.point_cloud_utils import transform_pcd_np


def get_valid_pixels_based_on_workspace(depth, intrinsic_matrix, T_WC):
    """
    Filter pixels that fall outside the robot's physical workspace boundaries.

    This function converts depth image pixels to 3D points in the world frame and checks
    which points fall within a predefined 3D bounding box representing the robot's
    reachable workspace. This is useful for:
    - Removing robot arm pixels if it enters the camera view
    - Filtering out background objects (walls, floor, ceiling)
    - Excluding objects outside the manipulation workspace

    Args:
        depth: Depth image (H, W) in millimeters (uint16) or meters (float)
        intrinsic_matrix: Camera intrinsic matrix (3, 3)
        T_WC: Camera extrinsics - transformation from camera to world frame (4, 4)

    Returns:
        Boolean mask (H, W) where True indicates pixels inside workspace bounds

    Workspace boundaries (in world frame, meters):
        - X (depth from robot base): 0.3m to 0.95m
        - Y (lateral left/right): -0.55m to 0.55m
        - Z (height above table): -0.07m to 0.45m

    Note: These boundaries are calibrated for a Sawyer robot with tabletop setup.
          Adjust these values for your specific robot and workspace configuration.
    """
    # Convert depth image to 3D point cloud in camera frame
    pcd_camera = img_to_pcd(depth, intrinsic_matrix, return_only_valid=False)

    # Transform point cloud to world frame
    pcd_world = transform_pcd_np(pcd_camera, T_WC, side='left')

    # Reshape back to image dimensions for per-pixel filtering
    pcd_img = pcd_world.reshape(depth.shape[0], depth.shape[1], 3)

    # Z-axis validation (height above table surface)
    above_min_z = pcd_img[..., 2] > -0.07  # Slightly below table surface
    below_max_z = pcd_img[..., 2] < 0.45   # Reasonable height for tabletop objects
    z_valid = np.logical_and(above_min_z, below_max_z)

    # X-axis validation (depth from robot base)
    not_too_close = pcd_img[..., 0] > 0.3   # Not too close to robot
    not_too_far = pcd_img[..., 0] < 0.95    # Within reach
    x_valid = np.logical_and(not_too_close, not_too_far)

    # Y-axis validation (lateral workspace bounds)
    not_too_far_to_left = pcd_img[..., 1] > -0.55
    not_too_far_to_right = pcd_img[..., 1] < 0.55
    y_valid = np.logical_and(not_too_far_to_left, not_too_far_to_right)

    # Combine all constraints
    xy_valid = np.logical_and(x_valid, y_valid)
    xyz_valid = np.logical_and(xy_valid, z_valid)

    return xyz_valid


def drawContour(m, s, c, RGB):
    """
    Draw edges of contour 'c' from segmented image 's' onto 'm' in colour 'RGB'.

    Helper function for visualizing instance segmentation boundaries.
    """
    # Fill contour "c" with white, make all else black
    thisContour = s.point(lambda p: p == c and 255)

    # Find edges of this contour and make into Numpy array
    thisEdges = thisContour.filter(ImageFilter.FIND_EDGES)
    thisEdgesN = np.array(thisEdges)

    # Paint locations of found edges in color "RGB" onto "main"
    m[np.nonzero(thisEdgesN)] = RGB
    return m


def overlay_instance_segmaps(rgb, instance_segmaps):
    """
    Overlay instance segmentation boundaries on RGB image.

    Args:
        rgb: RGB image (H, W, 3) uint8
        instance_segmaps: List of boolean masks, one per instance

    Returns:
        Desaturated RGB image with colored contours for each instance
    """
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    # Desaturate and revert to RGB, so we can draw on it in colour
    main = Image.fromarray(rgb).convert('L').convert('RGB')
    mainN = np.array(main)

    for i, seg in enumerate(instance_segmaps):
        seg = Image.fromarray(seg.cpu().numpy()).convert('L')
        mainN = drawContour(mainN, seg, 255, colours[i % 6])

    return mainN
