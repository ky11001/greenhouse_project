from typing import Tuple, Union

import numpy as np

RGB_Image = np.ndarray
DEPTH_Image = np.ndarray
Image = Union[RGB_Image, DEPTH_Image]
ImgSize = Tuple[int, int]


def resize_img(img: Image, new_size: ImgSize, img_type: str = 'rgb'):
    import cv2  # Moved inside to be able to debug using pycharm
    if img_type == 'rgb':
        return cv2.resize(img, new_size)
    elif img_type == 'depth':
        return cv2.resize(img, new_size, 0, 0, interpolation=cv2.INTER_NEAREST)
    elif img_type == 'segmap':
        return cv2.resize(img.astype(np.uint8), new_size, 0, 0, interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        raise Exception(f"The image type {img_type} is not supported. \n \
                          choose between 'rgb' and 'depth'")
