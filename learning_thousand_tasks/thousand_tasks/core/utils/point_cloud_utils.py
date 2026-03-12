import numpy as np
import open3d as o3d


def transform_pcd_np(pcd, T, side='left'):
    pcd_h = np.ones((pcd.shape[0], 4))
    pcd_h[:, :3] = pcd
    if side == 'left':
        pcd_h = np.matmul(T, pcd_h.T).T
    elif side == 'right':
        pcd_h = np.matmul(pcd_h, T.T)
    return pcd_h[:, :3]


def remove_separate_clusters(pcd, min_num_points=400):
    # Filter out disconnected points from the pcd using DBSCAN from open3d.
    # first convert to open3d point cloud
    pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    # Then apply DBSCAN
    labels = np.array(pcd_o3d.cluster_dbscan(eps=0.01, min_points=10, print_progress=False))

    # remove clusters with few points
    unique, counts = np.unique(labels, return_counts=True)
    if unique[0] == -1:
        unique = unique[1:]
        counts = counts[1:]

    # My implementation
    valid_clusters = unique[counts > min_num_points]
    main_pt_indices = np.array(np.where(np.in1d(labels, valid_clusters) == True))  # Dp not change ==

    # # Vitalis' implementation
    # pt_num = unique[np.argmax(counts)]
    # main_pt_indices = np.where(labels == pt_num)
    # main_pt_indices = np.array(main_pt_indices)

    # pcd_numpy = np.asarray(pcd_o3d.points)
    # segmented_pts = np.array(pcd_numpy[main_pt_indices[0, :], :])

    return main_pt_indices[0, :]


def backproject_camera_target_realworld(im_depth, rgb, K, target_mask=None):
    Kinv = np.linalg.inv(K)

    width = im_depth.shape[1]
    height = im_depth.shape[0]
    depth = im_depth.astype(np.float32, copy=True).flatten()
    rgb = rgb.astype(np.float32, copy=True).reshape(-1, 3)

    if target_mask is not None:
        mask = (depth != 0) * (target_mask.flatten() == 0)
    else:
        mask = (depth != 0)

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)  # each pixel

    # backprojection
    R = Kinv.dot(x2d.transpose())  #
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)

    return X[:, mask].transpose(), rgb[mask]


def find_most_occurring_point_cloud(labels, except_index=-2):
    cur_num = -10
    max_occurrence = -10
    temp_occurrence = 0
    max_num = -1
    for l in labels:
        if l != except_index and l != -1:
            if l == cur_num:
                temp_occurrence += 1
            else:
                cur_num = l
                temp_occurrence = 1
            if temp_occurrence > max_occurrence:
                max_occurrence = temp_occurrence
                max_num = l
    return max_num


def voxel_downsample_pcd(pcd_np, voxel_size=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    pcd_new = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd_new.points)


def remove_small_clusters_from_pcd(pcd, pcd_rgb, min_num_points=300):
    # Subsample. should be same as during inference (get_action).
    remain_indices = remove_separate_clusters(pcd, min_num_points=min_num_points)
    pcd = pcd[remain_indices]
    pcd_rgb = pcd_rgb[remain_indices]

    return pcd, pcd_rgb


def downsample_pcd(pcd, pcd_rgb, max_number_points=2048, replace=None):

    if replace is None:
        replace = True if len(pcd) < max_number_points else False
    rand_idx = np.random.choice(len(pcd), max_number_points, replace=replace)
    pcd = pcd[rand_idx]
    pcd_rgb = pcd_rgb[rand_idx]

    return pcd, pcd_rgb
