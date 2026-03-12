import re
import inspect
from os.path import join, exists
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Slerp, Rotation as Rot

from thousand_tasks.core.globals import ASSETS_DIR
from thousand_tasks.core.utils.scene_state import SceneState


def load_demo_scene_state(task_name: str,
                          segmentator=None,
                          segmentation_prompt_dict=None,
                          load_segmap_if_exists=True,
                          learned_tasks_dir=None):
    """Load demonstration scene state from saved files."""
    if learned_tasks_dir is None:
        learned_tasks_dir = ASSETS_DIR / 'demonstrations'

    task_dir = Path(learned_tasks_dir) / task_name

    rgb = np.asarray(Image.open(str(task_dir / 'head_camera_ws_rgb.png')))
    depth = np.load(str(task_dir / 'head_camera_ws_depth_to_rgb.npy'))[0]
    segmap = np.load(str(task_dir / 'head_camera_ws_segmap.npy'))[0]
    intrinsic_matrix = np.load(str(task_dir / 'head_camera_rgb_intrinsic_matrix.npy'))

    segmap_path = task_dir / 'head_camera_ws_segmap.npy'
    if segmap_path.exists() and load_segmap_if_exists:
        segmap = np.load(str(segmap_path))
        scene_state = SceneState.initialise_from_dict({
            'rgb': rgb,
            'depth': depth,
            'segmap': segmap,
            'intrinsic_matrix': intrinsic_matrix
        })
    else:
        scene_state = SceneState.initialise_from_dict({
            'rgb': rgb,
            'depth': depth,
            'intrinsic_matrix': intrinsic_matrix
        })

    return scene_state


def remove_demo_number_from_task_folder_name(task_folder_name):
    """Remove trailing demo number (e.g., _001) from task folder name."""
    if re.search(r'\d{3}', task_folder_name.split('_')[-1]) is not None:
        task_folder_name = '_'.join(task_folder_name.split('_')[:-1])
    return task_folder_name


def get_skill_name(task_dir):
    """Load skill name from demonstration directory."""
    skill_file = Path(task_dir) / "skill_name.txt"
    if skill_file.exists():
        with open(str(skill_file), "r") as f:
            return f.read().split('\n')[0]
    # Fallback: use task_description.txt
    task_desc_file = Path(task_dir) / "task_description.txt"
    if task_desc_file.exists():
        with open(str(task_desc_file), "r") as f:
            return f.read().split('\n')[0]
    raise FileNotFoundError(f"No skill_name.txt or task_description.txt found in {task_dir}")


def interpolate_poses(starting_pose, target_pose, num_interpolations):
    """Interpolate between two SE(3) poses using linear translation and SLERP rotation."""
    trans_actions = np.linspace(starting_pose[:3, 3], target_pose[:3, 3], num_interpolations)
    slerp = Slerp([0, 1], Rot.from_matrix(np.stack((starting_pose[:3, :3], target_pose[:3, :3]))))
    rot_actions = slerp(np.linspace(0, 1, num_interpolations)).as_rotvec()

    pose_actions = []
    for k in range(num_interpolations):
        pose = np.eye(4)
        pose[:3, 3] = trans_actions[k]
        pose[:3, :3] = Rot.from_rotvec(rot_actions[k]).as_matrix()
        pose_actions.append(pose)

    return pose_actions


def printarr(*arrs, float_width=6):
    """Print arrays with name, shape, dtype, and statistics in a formatted table."""
    frame = inspect.currentframe().f_back
    default_name = "[temporary]"

    def name_from_outer_scope(a):
        if a is None:
            return '[None]'
        name = default_name
        for k, v in frame.f_locals.items():
            if v is a:
                name = k
                break
        return name

    def dtype_str(a):
        if a is None:
            return 'None'
        if isinstance(a, int):
            return 'int'
        if isinstance(a, float):
            return 'float'
        return str(a.dtype)

    def shape_str(a):
        if a is None:
            return 'N/A'
        if isinstance(a, (int, float)):
            return 'scalar'
        return str(list(a.shape))

    def type_str(a):
        return str(type(a))[8:-2]

    def device_str(a):
        if hasattr(a, 'device'):
            device_str = str(a.device)
            if len(device_str) < 10:
                return device_str
        return ""

    def format_float(x):
        return f"{x:{float_width}g}"

    def minmaxmean_str(a):
        if a is None:
            return ('N/A', 'N/A', 'N/A')
        if isinstance(a, (int, float)):
            return (format_float(a), format_float(a), format_float(a))

        min_str = max_str = mean_str = "N/A"
        try:
            min_str = format_float(a.min())
        except:
            pass
        try:
            max_str = format_float(a.max())
        except:
            pass
        try:
            mean_str = format_float(a.mean())
        except:
            pass

        return (min_str, max_str, mean_str)

    try:
        props = ['name', 'dtype', 'shape', 'type', 'device', 'min', 'max', 'mean']

        str_props = []
        for a in arrs:
            minmaxmean = minmaxmean_str(a)
            str_props.append({
                'name': name_from_outer_scope(a),
                'dtype': dtype_str(a),
                'shape': shape_str(a),
                'type': type_str(a),
                'device': device_str(a),
                'min': minmaxmean[0],
                'max': minmaxmean[1],
                'mean': minmaxmean[2],
            })

        maxlen = {p: 0 for p in props}
        for sp in str_props:
            for p in props:
                maxlen[p] = max(maxlen[p], len(sp[p]))

        props = [p for p in props if maxlen[p] > 0]
        maxlen = {p: max(maxlen[p], len(p)) for p in props}

        header_str = ""
        for p in props:
            prefix = "" if p == 'name' else " | "
            fmt_key = ">" if p == 'name' else "<"
            header_str += f"{prefix}{p:{fmt_key}{maxlen[p]}}"
        print(header_str)
        print("-" * len(header_str))

        for strp in str_props:
            for p in props:
                prefix = "" if p == 'name' else " | "
                fmt_key = ">" if p == 'name' else "<"
                print(f"{prefix}{strp[p]:{fmt_key}{maxlen[p]}}", end='')
            print("")

    finally:
        del frame
