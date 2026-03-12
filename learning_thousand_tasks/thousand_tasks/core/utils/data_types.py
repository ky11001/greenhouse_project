from typing import Tuple

import numpy as np
import torch

ObjID = int
RGB_Image_numpy = np.ndarray
RGB_Image_torch = torch.Tensor

DEPTH_Image_numpy = np.ndarray

Instance_Segmap_torch = torch.Tensor  # Image size, all pixels are labeled with the index of the object they correspond to

Segmap_numpy = np.ndarray  # Binary segmentation mask for a single object
Segmap_torch = torch.Tensor  # Binary segmentation mask for a single object
Segmaps_numpy = np.ndarray #Stacked Segmap_numpy

Segmap_Group_torch = torch.Tensor  # Group of binary segmentation masks (n_objects, height, width)

Cropped_Images_torch = torch.Tensor

ImgSize_height_width = Tuple[int, int]  # Image height, width
Bounding_Boxes = torch.Tensor  # Each row is (xmin, ymin, wifth, height) for an object
Clip_Encodings_numpy = np.ndarray
