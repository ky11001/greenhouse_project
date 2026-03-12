import numpy as np

from thousand_tasks.core.utils.img_to_pcd import img_to_o3d_pcd
from thousand_tasks.core.utils.data_types import ImgSize_height_width
from thousand_tasks.core.utils.image import resize_img
from thousand_tasks.core.utils.intrinsic_matrix_utils import calculate_intrinsic_for_new_resolution, \
    calculate_intrinsic_matrix_for_crop


class SceneState:

    def __init__(self, pcd_remove_outliers=True):
        self._T_WC = None

        self._current_rgb = None
        self._current_depth = None
        self._current_segmap = None
        self._current_pcd_world = None

        self._obj_was_cropped = False
        self._width_obj_crop = None
        self._height_obj_crop = None
        self._intrinsic_matrix_obj_crop = None
        self._rgb_obj_crop = None
        self._depth_obj_crop = None
        self._segmap_obj_crop = None

        self._current_img_width = None
        self._current_img_height = None
        self._current_intrinsic_matrix = None
        self._pcd_remove_outliers = pcd_remove_outliers

    @staticmethod
    def initialise_from_dict(data: dict, pcd_remove_outliers=True):
        """
        The dictionary must have a key 'intrinsic_matrix'. In addition, it must also have a key 'rgb' and/or 'depth'.
        Optionally, one can also pass a 'segmap' key.
        """

        scene_state = SceneState(pcd_remove_outliers=pcd_remove_outliers)

        dict_keys = data.keys()
        assert (('rgb' in dict_keys) or ('depth' in dict_keys)) and (
                'intrinsic_matrix' in dict_keys), 'You must provide either an RGB image and intrinsic matrix or depth image and intrinsic matrix or both'
        if 'rgb' in dict_keys and 'depth' in dict_keys:
            assert data['rgb'].shape[:2] == data[
                'depth'].shape, f"RGB image must have the same dimensionality as the depth image. Current dimensionality are {data['rgb'].shape[:2]} and {data['depth'].shape}"

        if 'rgb' in dict_keys:
            scene_state.rgb = data['rgb']
            scene_state.img_size = (data['rgb'].shape[0], data['rgb'].shape[1])

        if 'depth' in dict_keys:
            scene_state.depth = data['depth']
            scene_state.img_size = (data['depth'].shape[0], data['depth'].shape[1])

        if 'segmap' in dict_keys:
            assert (data['segmap'].shape[0], data['segmap'].shape[1]) == scene_state.img_size, \
                f"Segmap size {(data['segmap'].shape[0], data['segmap'].shape[1])} does not match stored image size {scene_state.img_size}"
            scene_state.segmap = data['segmap']

        scene_state.intrinsic_matrix = data['intrinsic_matrix']

        return scene_state

    def resize_images(self,
                      new_height: int,
                      new_width: int):

        current_img_size = self.img_size
        self.img_size = (new_height, new_width)

        if self.rgb_was_set:
            assert self.rgb_was_set, 'RGB image was not set'
            self.rgb = resize_img(img=self.rgb, new_size=(new_width, new_height), img_type='rgb')

        if self.depth_was_set:
            assert self.depth_was_set, 'Depth image was not set'
            self.depth = resize_img(img=self.depth, new_size=(new_width, new_height), img_type='depth')

        if self.segmap_was_set:
            assert self.segmap_was_set, 'Segmap image was not set'
            self.segmap = resize_img(img=self.segmap, new_size=(new_width, new_height), img_type='segmap')

        self._modify_intrinsic_matrix_after_resizing(old_size=current_img_size, new_size=self.img_size)

    def crop_images(self,
                    y_min: int,
                    y_max: int,
                    x_min: int,
                    x_max: int):

        assert x_max > x_min
        assert y_max > y_min
        self.img_size = (y_max - y_min, x_max - x_min)

        if self.rgb_was_set:
            assert self.rgb_was_set, 'RGB image was not set'
            self.rgb = self.rgb[y_min:y_max, x_min:x_max, :]

        if self.depth_was_set:
            assert self.depth_was_set, 'Depth image was not set'
            self.depth = self.depth[y_min:y_max, x_min:x_max]

        if self.segmap_was_set:
            assert self.segmap_was_set, 'Segmap image was not set'
            self.segmap = self.segmap[y_min:y_max, x_min:x_max]

        self._modify_intrinsic_matrix_after_cropping(y_min=y_min, x_min=x_min)

    def crop_object(self, x_min, y_min, width, height):
        self._obj_was_cropped = True

        if self.rgb_was_set:
            self._rgb_obj_crop = self.rgb[y_min:y_min + height, x_min:x_min + width]
        if self.depth_was_set:
            self._depth_obj_crop = self.depth[y_min:y_min + height, x_min:x_min + width]
        if self.segmap_was_set:
            self._segmap_obj_crop = self.segmap[y_min:y_min + height, x_min:x_min + width]

        self._width_obj_crop = width
        self._height_obj_crop = height
        self._intrinsic_matrix_obj_crop = calculate_intrinsic_matrix_for_crop(intrinsic_matrix=self.intrinsic_matrix,
                                                                              y_min=y_min,
                                                                              x_min=x_min)

    def resize_obj_crops(self,
                         new_height: int,
                         new_width: int):
        assert self.obj_was_cropped, 'Object was not cropped.'

        initial_crop_height, initial_crop_width = self._height_obj_crop, self._width_obj_crop

        if self.rgb_was_set:
            self._rgb_obj_crop = resize_img(img=self.obj_crop_rgb, new_size=(new_width, new_height), img_type='rgb')

        if self.depth_was_set:
            self._depth_obj_crop = resize_img(img=self.obj_crop_depth, new_size=(new_width, new_height),
                                              img_type='depth')

        if self.segmap_was_set:
            self._segmap_obj_crop = resize_img(img=self.obj_crop_segmap, new_size=(new_width, new_height),
                                               img_type='segmap')

        self._height_obj_crop = new_height
        self._width_obj_crop = new_width
        self._intrinsic_matrix_obj_crop = calculate_intrinsic_for_new_resolution(
            intrinsic_matrix=self._intrinsic_matrix_obj_crop,
            new_width=new_width,
            new_height=new_height,
            old_width=initial_crop_width,
            old_height=initial_crop_height)

    def crop_object_using_segmap(self, margin: float = 0.5):
        y_min, x_min, y_max, x_max = self._get_bouding_box(margin)
        height = y_max - y_min
        width = x_max - x_min
        self.crop_object(x_min=x_min, y_min=y_min, width=width, height=height)

    def _get_bouding_box(self, margin: float = 0.5):
        assert self.segmap_was_set, 'Can\'t calculate a bounding box without a segmap'
        y, x = np.where(self.segmap != 0)
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)

        height = y_max - y_min
        width = x_max - x_min

        width_margin = np.round(width * (margin / 2))
        height_margin = np.round(height * (margin / 2))

        x_min = max(x_min - width_margin, 0)
        x_max = min(x_max + width_margin, self.img_width)

        y_min = max(y_min - height_margin, 0)
        y_max = min(y_max + height_margin, self.img_height)

        return int(y_min), int(x_min), int(y_max), int(x_max)

    @property
    def T_WC(self):
        if self.T_WC_was_set:
            return self._T_WC.copy()
        else:
            raise Exception('The extrinsic matrix has not been set.')

    @T_WC.setter
    def T_WC(self, T_WC):
        assert len(
            T_WC.shape) == 2, f'T_WC should have 2 dimensions and not {len(T_WC.shape)}'
        assert T_WC.dtype in [np.float32, np.float64,
                              float], f'T_WC should have dtype float and not {T_WC.dtype}'
        assert (T_WC.shape[0], T_WC.shape[1]) == (4,
                                                  4), f'T_WC dimensions do not match. Expected {(4, 4)} while received {(T_WC.shape[0], T_WC.shape[1])}'

        self._T_WC = T_WC

    @property
    def intrinsic_matrix(self):
        if self.intrinsic_matrix_was_set:
            return self._current_intrinsic_matrix.copy()
        else:
            raise Exception('The intrinsic matrix has not been set.')

    @intrinsic_matrix.setter
    def intrinsic_matrix(self, intrinsic_matrix):
        assert len(
            intrinsic_matrix.shape) == 2, f'The intrinsic matrix should have 2 dimensions and not {len(intrinsic_matrix.shape)}'
        assert intrinsic_matrix.dtype in [np.float32, np.float64,
                                          float], f'The intrinsic matrix should have dtype float and not {intrinsic_matrix.dtype}'
        assert (intrinsic_matrix.shape[0], intrinsic_matrix.shape[1]) == (3,
                                                                          3), f'Intrinsic matrix dimensions do not match. Expected {(3, 3)} while received {(intrinsic_matrix.shape[0], intrinsic_matrix.shape[1])}'

        self._current_intrinsic_matrix = intrinsic_matrix

    @property
    def img_width(self):
        if self.img_width_was_set:
            return self._current_img_width
        else:
            raise Exception('Image width has not been set.')

    @img_width.setter
    def img_width(self, img_width):
        assert isinstance(img_width, int), 'img_width should be an integer'

        self._current_img_width = img_width

    @property
    def img_height(self):
        if self.img_height_was_set:
            return self._current_img_height
        else:
            raise Exception('Image width has not been set.')

    @img_height.setter
    def img_height(self, img_height):
        assert isinstance(img_height, int), 'img_height should be an integer'

        self._current_img_height = img_height

    @property
    def img_size(self):
        if self.img_width_was_set and self.img_height_was_set:
            return (self.img_height, self.img_width)

    @img_size.setter
    def img_size(self, size: ImgSize_height_width):
        self.img_height = size[0]
        self.img_width = size[1]

    @property
    def obj_crop_img_height(self):
        if self.obj_was_cropped:
            return self._height_obj_crop

    @property
    def obj_crop_img_width(self):
        if self.obj_was_cropped:
            return self._width_obj_crop

    @property
    def obj_crop_img_size(self):
        if self.obj_was_cropped:
            return (self.obj_crop_img_height, self.obj_crop_img_width)

    @property
    def rgb(self):
        if self.rgb_was_set:
            return self._current_rgb.copy()

    @rgb.setter
    def rgb(self, rgb):
        assert len(rgb.shape) == 3, f'RGB images should have 3 dimensions and not {len(rgb.shape)}'
        assert rgb.shape[2] == 3, f'The last channel should have 3 dimensions and not {rgb.shape[2]}'
        assert rgb.dtype == np.uint8, f'RGB images should have dtype uint8 and not {rgb.dtype}'
        if self.img_size is not None:
            assert (rgb.shape[0], rgb.shape[1]) == \
                   (self.img_height,
                    self.img_width), f'Image dimensions do not match. Expected {(self.img_height, self.img_width)} while received {(rgb.shape[0], rgb.shape[1])}'
        else:
            self.img_size = (rgb.shape[0], rgb.shape[1])

        self._current_rgb = rgb

    @property
    def depth(self):
        if self.depth_was_set:
            return self._current_depth.copy()

    @depth.setter
    def depth(self, depth):
        assert len(depth.shape) == 2, f'Depth images should have 2 dimensions and not {len(depth.shape)}'
        assert depth.dtype in [np.uint16,
                               np.uint32, np.int16,
                               np.int32], f'Depth images should have dtype uint16 or uint32 and not {depth.dtype}'
        if self.img_size is not None:
            assert (depth.shape[0], depth.shape[1]) == \
                   (self.img_height,
                    self.img_width), f'Image dimensions do not match. Expected {(self.img_height, self.img_width)} while received {(depth.shape[0], depth.shape[1])}'

        else:
            self.img_size = (depth.shape[0], depth.shape[1])

        self._current_depth = depth.astype(np.uint16)

    @property
    def segmap(self):
        if self.segmap_was_set:
            return self._current_segmap.copy()

    @segmap.setter
    def segmap(self, segmap):
        assert len(segmap.shape) == 2, f'Segmentation masks should have 2 dimensions and not {len(segmap.shape)}'
        assert segmap.dtype == bool, f'Segmentation masks  should have dtype bool and not {segmap.dtype}'
        assert (segmap.shape[0], segmap.shape[1]) == (self.img_height,
                                                      self.img_width), f'Image dimensions do not match. Expected {(self.img_height, self.img_width)} while received {(segmap.shape[0], segmap.shape[1])}'

        self._current_segmap = segmap

    def erode_segmap(self):
        import cv2  # Moved inside to be able to debug using pycharm
        assert self.segmap_was_set, 'Segmentation mask was not set'
        kernel = np.ones((3, 3), dtype=np.float32)
        kernel = kernel / kernel.sum()
        segmap = self.segmap.astype(np.float32)
        for _ in range(1):
            segmap = cv2.filter2D(src=segmap, ddepth=-1, kernel=kernel)
            segmap = (segmap >= 1)
        self._current_segmap = segmap

    @property
    def o3d_pcd(self):
        if self.depth_was_set and self.intrinsic_matrix_was_set:
            if self.segmap is None:
                pcd = img_to_o3d_pcd(depth=self.depth,
                                     intrinsic_matrix=self.intrinsic_matrix,
                                     rgb=self.rgb)
            else:
                # depth = self.depth
                # seg = self.segmap
                #
                # n_smallest = 5000
                # std_scale = 0.6
                # flat_depth = depth.reshape(-1)
                # seg_flat_depth = depth.reshape(-1)[seg.reshape(-1)]
                #
                # semi_sorted_args = np.argpartition(-seg_flat_depth[:], n_smallest)  # get arg of 50 largest values
                # largest_values = seg_flat_depth[semi_sorted_args[:n_smallest]]
                #
                # mean, std = np.mean(largest_values), np.std(largest_values)
                # filter_args = flat_depth > (mean + std_scale * std)
                #
                # filtered_segmap = filter_args.reshape((self.depth.shape[0], self.depth.shape[1]))
                # filtered_segmap = (filtered_segmap * self.segmap).astype(bool)
                # self.segmap = filtered_segmap * self.segmap
                # self.crop_object_using_segmap()

                pcd = img_to_o3d_pcd(depth=self.depth * self.segmap,
                                     intrinsic_matrix=self.intrinsic_matrix,
                                     rgb=self.rgb * self.segmap[..., None])

                # pcd = img_to_o3d_pcd(depth=self.depth * self.segmap,
                #                      intrinsic_matrix=self.intrinsic_matrix,
                #                      rgb=self.rgb * self.segmap[..., None])

            pcd = pcd.voxel_down_sample(0.002)
            if self._pcd_remove_outliers:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.8)

            # pcd, _ = pcd.remove_radius_outlier(nb_points=20, radius=0.01)
            return pcd
        else:
            raise Exception(
                f'It is not possible to create a point cloud. Depth was set: {self.depth_was_set}, intrinsic matrix was set: {self.intrinsic_matrix_was_set}')
        
    @property
    def pcd_world(self):
        if self.pcd_world_was_set:
            return self._current_pcd_world.copy()

    @pcd_world.setter
    def pcd_world(self, pcd: np.ndarray):
        assert len(pcd.shape) == 2, f'Pointcloud should have 2 dimensions and not {len(pcd.shape)}'
        assert pcd.shape[1] == 3, f'Each point of pointcloud \
            should have 3 dimensions (x, y, z) and not {pcd.shape[1]}'
        self._current_pcd_world = pcd

    @property
    def obj_crop_rgb(self):
        if self.obj_was_cropped and self.rgb_was_set:
            return self._rgb_obj_crop
        else:
            raise Exception(f'RGB was set: {self.rgb_was_set}. Object was cropped: {self.obj_was_cropped}')

    @property
    def obj_crop_depth(self):
        if self.obj_was_cropped and self.depth_was_set:
            return self._depth_obj_crop
        else:
            raise Exception(f'Depth was set: {self.depth_was_set}. Object was cropped: {self.obj_was_cropped}')

    @property
    def obj_crop_segmap(self):
        if self.obj_was_cropped and self.segmap_was_set:
            return self._segmap_obj_crop
        else:
            raise Exception(f'Segmap was set: {self.segmap}. Object was cropped: {self.obj_was_cropped}')

    @property
    def obj_crop_intrinsic_matrix(self):
        if self.obj_was_cropped:
            return self._intrinsic_matrix_obj_crop
        else:
            raise Exception(f'Object was cropped: {self.obj_was_cropped}')

    def _modify_intrinsic_matrix_after_cropping(self, y_min: int, x_min: int):
        self.intrinsic_matrix = calculate_intrinsic_matrix_for_crop(intrinsic_matrix=self.intrinsic_matrix,
                                                                    y_min=y_min,
                                                                    x_min=x_min)

    def _modify_intrinsic_matrix_after_resizing(self, old_size: ImgSize_height_width, new_size: ImgSize_height_width):
        self.intrinsic_matrix = calculate_intrinsic_for_new_resolution(intrinsic_matrix=self.intrinsic_matrix,
                                                                       new_width=new_size[1],
                                                                       new_height=new_size[0],
                                                                       old_width=old_size[1],
                                                                       old_height=old_size[0])

    @property
    def obj_was_cropped(self):
        return self._obj_was_cropped

    @property
    def rgb_was_set(self):
        return self._current_rgb is not None

    @property
    def depth_was_set(self):
        return self._current_depth is not None

    @property
    def segmap_was_set(self):
        return self._current_segmap is not None
    
    @property
    def pcd_world_was_set(self):
        return self._current_pcd_world is not None

    @property
    def intrinsic_matrix_was_set(self):
        return self._current_intrinsic_matrix is not None

    @property
    def img_width_was_set(self):
        return self._current_img_width is not None

    @property
    def img_height_was_set(self):
        return self._current_img_height is not None

    @property
    def T_WC_was_set(self):
        return self._T_WC is not None
