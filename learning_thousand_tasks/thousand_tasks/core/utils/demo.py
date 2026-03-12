import os
import shutil
from os.path import exists, join

import imageio
import numpy as np
from PIL import Image, ImageOps

from thousand_tasks.core.globals import TASKS_DIR


def make_dir(dir_path, delete_if_exists=False):
    try:
        os.mkdir(dir_path)
    except:
        if delete_if_exists:
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)

        pass


def check_if_to_collect_demo(task_name, task_dir):
    if os.path.exists(task_dir):
        message = f'\nDirectory for task \'{task_name}\' already exists. Overwrite directory (\'y\') or proceed to next task (\'n\')?'
        user_choice = input(message)
        if user_choice.lower() == 'y' or user_choice.lower() == 'yes':
            return True
        elif user_choice.lower() == 'n' or user_choice.lower() == 'no':
            return False
        else:
            print(f'{user_choice} is not a valid answer. Please choose from: \'y\' and \'n\' for yes and no')
            return check_if_to_collect_demo(task_dir)
    else:
        return True


def check_if_demo_was_successful():
    message = f'\nWas the demonstration successful (y/n)?'
    user_choice = input(message)
    if user_choice.lower() == 'y' or user_choice.lower() == 'yes':
        return True
    elif user_choice.lower() == 'n' or user_choice.lower() == 'no':
        return False
    else:
        print(f'{user_choice} is not a valid answer. Please choose from: \'y\' and \'n\' for yes and no')
        return check_if_demo_was_successful()


class Demo:

    def __init__(self, task_name, tasks_dir=TASKS_DIR):
        self.task_name = task_name
        self.tasks_dir = tasks_dir
        self.task_dir_path = join(self.tasks_dir, self.task_name)
        assert exists(
            self.task_dir_path), f'Task directory for task {self.task_name} can\'t be found under path {self.task_dir_path}'

        self.bottleneck_pose = np.load(join(self.task_dir_path, 'bottleneck_pose.npy'))
        self.bottleneck_posevec = np.load(join(self.task_dir_path, 'bottleneck_posevec.npy'))
        self.demo_eef_posevecs = np.load(join(self.task_dir_path, 'demo_eef_posevecs.npy'))
        self.demo_eef_twists = np.load(join(self.task_dir_path, 'demo_eef_twists.npy'))

        self.rgb_images = {}
        self.depth_images = {}
        self.intrinsic_matrices = {}

        self.workspace_rgb_images = {}
        self.workspace_depth_images = {}
        self.bottleneck_rgb_images = {}
        self.bottleneck_depth_images = {}

        self.segmented_rgb_images = {}
        self.segmented_depth_images = {}

    def load_rgbd_images(self, camera_name='external_camera'):
        self.rgb_images[camera_name] = np.load(join(self.task_dir_path, f'{camera_name}_rgb.npy'))
        self.depth_images[camera_name] = np.load(join(self.task_dir_path, f'{camera_name}_depth_to_rgb.npy'))
        self.intrinsic_matrices[camera_name] = np.load(
            join(self.task_dir_path, f'{camera_name}_rgb_intrinsic_matrix.npy'))

    def load_workspace_images(self, camera_name='external_camera'):
        self.workspace_rgb_images[camera_name] = np.asarray(
            Image.open(join(self.task_dir_path, f'{camera_name}_ws_rgb.png')))
        self.workspace_depth_images[camera_name] = np.asarray(
            Image.open(join(self.task_dir_path, f'{camera_name}_ws_depth_to_rgb.png')))

    def load_bottleneck_images(self, camera_name='external_camera'):
        self.bottleneck_rgb_images[camera_name] = np.asarray(
            Image.open(join(self.task_dir_path, f'{camera_name}_bottleneck_rgb.png')))
        self.bottleneck_depth_images[camera_name] = np.asarray(
            Image.open(join(self.task_dir_path, f'{camera_name}_bottleneck_depth_to_rgb.png')))

    def load_segmented_rgb_images(self, camera_name='external_camera'):
        segmaps = np.load(join(self.task_dir_path, f'{camera_name}_masks.npy'))
        rgb_images = np.load(join(self.task_dir_path, f'{camera_name}_rgb.npy'))
        depth_images = np.load(join(self.task_dir_path, f'{camera_name}_depth_to_rgb.npy'))

        self.segmented_rgb_images[camera_name] = rgb_images * segmaps[..., None]
        self.segmented_depth_images[camera_name] = depth_images * segmaps

    def create_gif_from_rgb_images(self, output_file_path, camera_name='external_camera', target_res=None, fps=30):
        assert output_file_path.split('.')[-1] == 'gif', 'Output file path extension should be \'.gif\''
        if target_res is not None:
            assert isinstance(target_res, tuple) and len(target_res) == 2
        frames = self.rgb_images[camera_name]
        if target_res is not None:
            frames = np.asarray(
                [np.asarray(ImageOps.contain(Image.fromarray(frame), size=target_res)) for frame in frames])

        imageio.mimsave(output_file_path, frames, fps=fps)

    def create_gif_from_segmented_rgb_images(self, output_file_path, camera_name='external_camera', target_res=None,
                                             fps=30):
        assert output_file_path.split('.')[-1] == 'gif', 'Output file path extension should be \'.gif\''
        if target_res is not None:
            assert isinstance(target_res, tuple) and len(target_res) == 2
        frames = self.segmented_rgb_images[camera_name]
        if target_res is not None:
            frames = np.asarray(
                [np.asarray(ImageOps.contain(Image.fromarray(frame), size=target_res)) for frame in frames])

        imageio.mimsave(output_file_path, frames, fps=fps)
