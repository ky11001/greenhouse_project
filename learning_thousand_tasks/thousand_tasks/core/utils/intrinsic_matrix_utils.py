import json
from typing import Tuple

import numpy as np
import open3d as o3d

ImgSize = Tuple[int, int]  # height, width


def calculate_intrinsic_for_new_resolution(intrinsic_matrix: np.ndarray, new_width, new_height, old_width, old_height):
    ratio_width = new_width / old_width
    ratio_height = new_height / old_height
    new_intrinsic = intrinsic_matrix.copy()
    new_intrinsic[0] *= ratio_width
    new_intrinsic[1] *= ratio_height
    return new_intrinsic


def intrinsic_matrix_to_o3d(intrinsic_matrix: np.ndarray, image_h=None, image_w=None):
    intrinsic_matrix_o3d = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_matrix_o3d.intrinsic_matrix = intrinsic_matrix

    if image_w is not None:
        intrinsic_matrix_o3d.width = image_w
    if image_h is not None:
        intrinsic_matrix_o3d.height = image_h

    return intrinsic_matrix_o3d


def calculate_intrinsic_matrix_for_crop(intrinsic_matrix: np.ndarray, y_min: int, x_min: int):
    x0, y0 = intrinsic_matrix[0:2, 2]
    x0 = x0 - x_min
    y0 = y0 - y_min
    intrinsic_matrix[0:2, 2] = [x0, y0]
    return intrinsic_matrix


def load_intrinsics(directory: str):
    return np.load(directory)


# TODO remove as not needed
def update_json(intrinsics: np.ndarray, img_size: ImgSize, path: str):
    params = {
        "img_width": img_size[0],
        "img_height": img_size[1],
        "fx": intrinsics[0, 0],
        "fy": intrinsics[1, 1],
        "x_offset": intrinsics[0, 2],
        "y_offset": intrinsics[1, 2]
    }
    with open(path, 'w') as params_file:
        json_params = json.dumps(params)
        params_file.write(json_params)
